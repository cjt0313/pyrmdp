#!/usr/bin/env python3
"""
analyze_consistency.py — Experiment B: within-task graph consistency analysis.

Reads saved intermediates from batch_evaluate_all.py (lifted trajectories,
synthesized graphs, evaluation results) and computes:

A. Vocabulary consistency (per task)
B. Structural consistency (per task)
C. Cross-trajectory transfer (within-task and cross-task)
D. Three-level ablation (raw → pruned → full)
E. Task separability gap and statistical tests

No VLM calls needed — purely offline analysis.

Usage
-----
  python scripts/analyze_consistency.py \
      --experiment-dir /inspire/qb-ilm/.../experiment_output \
      -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx

from pyrmdp.offline_validate.graph_evaluator import (
    load_abstract_graph,
    evaluate,
)

logger = logging.getLogger("pyrmdp.consistency")

LiftedState = Tuple[FrozenSet[str], FrozenSet[str]]


# ════════════════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════════════════

def _save_json(path: Path, data: Any) -> None:
    def _default(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    path.write_text(
        json.dumps(data, indent=2, default=_default, ensure_ascii=False),
        encoding="utf-8",
    )


def load_lifted_trajectory(path: Path) -> List[LiftedState]:
    """Load a lifted trajectory from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        (frozenset(s["true"]), frozenset(s["false"]))
        for s in data["states"]
    ]


def extract_predicates_from_pddl(pddl_path: Path) -> set:
    """Extract bare predicate names from a PDDL domain file."""
    text = pddl_path.read_text(encoding="utf-8")
    start = text.find("(:predicates")
    if start < 0:
        return set()
    depth, end = 0, start
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = text[start + len("(:predicates"):end]
    block = re.sub(r";;[^\n]*", "", block)
    preds = set()
    for tok in re.findall(r"\(([^)]+)\)", block):
        tok = tok.strip()
        if tok and tok[0] != ";":
            preds.add(tok.split()[0])
    return preds


def extract_actions_from_pddl(pddl_path: Path) -> set:
    """Extract action names from a PDDL domain file."""
    text = pddl_path.read_text(encoding="utf-8")
    return set(re.findall(r"\(:action\s+(\S+)", text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ════════════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════════════

def discover_episodes(experiment_dir: Path) -> Dict[str, List[Dict]]:
    """Discover all completed episodes grouped by task."""
    per_traj_dir = experiment_dir / "per_trajectory"
    if not per_traj_dir.exists():
        raise FileNotFoundError(f"per_trajectory directory not found: {per_traj_dir}")

    tasks: Dict[str, List[Dict]] = defaultdict(list)
    for task_dir in sorted(per_traj_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            metadata_path = ep_dir / "metadata.json"
            lifted_path = ep_dir / "lifted_trajectory.json"
            if metadata_path.exists() and lifted_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                tasks[task_dir.name].append({
                    "episode_id": ep_dir.name,
                    "episode_dir": ep_dir,
                    "metadata": metadata,
                })

    for task in tasks:
        tasks[task].sort(key=lambda x: x["episode_id"])

    return dict(tasks)


# ════════════════════════════════════════════════════════════════════
#  A. Vocabulary Consistency
# ════════════════════════════════════════════════════════════════════

def analyze_vocabulary(tasks: Dict[str, List[Dict]]) -> Dict:
    """Compute vocabulary consistency metrics per task."""
    results = {}
    for task_name, episodes in tasks.items():
        pred_sets = []
        action_sets = []
        for ep in episodes:
            ep_dir = ep["episode_dir"]
            domain_path = ep_dir / "pipeline_output" / "step0_domain_generated.pddl"
            if domain_path.exists():
                pred_sets.append(extract_predicates_from_pddl(domain_path))
                action_sets.append(extract_actions_from_pddl(domain_path))

        if len(pred_sets) < 2:
            results[task_name] = {"num_episodes": len(pred_sets), "skip": True}
            continue

        pred_jaccards = [jaccard(a, b) for a, b in combinations(pred_sets, 2)]
        action_jaccards = [jaccard(a, b) for a, b in combinations(action_sets, 2)]

        all_preds = set().union(*pred_sets)
        pred_freq = {}
        for p in all_preds:
            freq = sum(1 for ps in pred_sets if p in ps) / len(pred_sets)
            pred_freq[p] = freq

        task_shared = {p for p, f in pred_freq.items() if f >= 0.5}
        episode_specific = {p for p, f in pred_freq.items() if f < 0.5}

        results[task_name] = {
            "num_episodes": len(pred_sets),
            "predicate_jaccard_mean": float(np.mean(pred_jaccards)),
            "predicate_jaccard_std": float(np.std(pred_jaccards)),
            "action_jaccard_mean": float(np.mean(action_jaccards)),
            "action_jaccard_std": float(np.std(action_jaccards)),
            "total_unique_predicates": len(all_preds),
            "task_shared_predicates": sorted(task_shared),
            "task_shared_fraction": len(task_shared) / max(len(all_preds), 1),
            "episode_specific_predicates": sorted(episode_specific),
            "episode_specific_fraction": len(episode_specific) / max(len(all_preds), 1),
            "predicate_frequency": {p: round(f, 3) for p, f in sorted(pred_freq.items())},
        }

    return results


# ════════════════════════════════════════════════════════════════════
#  B. Structural Consistency
# ════════════════════════════════════════════════════════════════════

def analyze_structure(tasks: Dict[str, List[Dict]]) -> Dict:
    """Compute structural graph metrics per task."""
    results = {}
    for task_name, episodes in tasks.items():
        graph_stats = []
        for ep in episodes:
            ep_dir = ep["episode_dir"]
            try:
                G = load_abstract_graph(ep_dir / "pipeline_output")
            except FileNotFoundError:
                continue

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            n_scc = nx.number_strongly_connected_components(G)
            sources = sum(1 for n in G.nodes() if G.in_degree(n) == 0)
            sinks = sum(1 for n in G.nodes() if G.out_degree(n) == 0)

            graph_stats.append({
                "episode_id": ep["episode_id"],
                "nodes": n_nodes,
                "edges": n_edges,
                "scc_count": n_scc,
                "sources": sources,
                "sinks": sinks,
            })

        if len(graph_stats) < 2:
            results[task_name] = {"num_episodes": len(graph_stats), "skip": True}
            continue

        metrics = {}
        for key in ["nodes", "edges", "scc_count", "sources", "sinks"]:
            vals = [s[key] for s in graph_stats]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            cv = std / mean if mean > 0 else 0.0
            metrics[key] = {"mean": mean, "std": std, "cv": cv, "min": min(vals), "max": max(vals)}

        results[task_name] = {
            "num_episodes": len(graph_stats),
            "per_episode": graph_stats,
            "aggregate": metrics,
        }

    return results


# ════════════════════════════════════════════════════════════════════
#  C. Cross-Trajectory Transfer
# ════════════════════════════════════════════════════════════════════

def _eval_graph_on_trajectory(
    graph_dir: Path,
    lifted_traj: List[LiftedState],
) -> Dict[str, float]:
    """Evaluate a single graph against a lifted trajectory."""
    try:
        G = load_abstract_graph(graph_dir)
    except FileNotFoundError:
        return {"state_recall": 0.0, "path_recall": 0.0, "error": "no_graph"}

    if not lifted_traj:
        return {"state_recall": 0.0, "path_recall": 0.0, "error": "empty_trajectory"}

    result = evaluate(lifted_traj, G)
    return {
        "state_recall": result.state_recall,
        "path_recall": result.path_recall,
    }


def compute_within_task_transfer(
    tasks: Dict[str, List[Dict]],
    variant: str = "pipeline_output",
) -> Dict:
    """Compute within-task N×N transfer matrices."""
    results = {}
    for task_name, episodes in tasks.items():
        n = len(episodes)
        if n < 2:
            results[task_name] = {"num_episodes": n, "skip": True}
            continue

        sr_matrix = np.zeros((n, n))
        pr_matrix = np.zeros((n, n))

        lifted_trajs = {}
        for i, ep in enumerate(episodes):
            path = ep["episode_dir"] / "lifted_trajectory.json"
            lifted_trajs[i] = load_lifted_trajectory(path)

        for i in range(n):
            graph_dir = episodes[i]["episode_dir"] / variant
            for j in range(n):
                metrics = _eval_graph_on_trajectory(graph_dir, lifted_trajs[j])
                sr_matrix[i, j] = metrics["state_recall"]
                pr_matrix[i, j] = metrics["path_recall"]

        diag_sr = float(np.mean(np.diag(sr_matrix)))
        diag_pr = float(np.mean(np.diag(pr_matrix)))

        mask = ~np.eye(n, dtype=bool)
        offdiag_sr = float(np.mean(sr_matrix[mask])) if n > 1 else 0.0
        offdiag_pr = float(np.mean(pr_matrix[mask])) if n > 1 else 0.0
        offdiag_sr_std = float(np.std(sr_matrix[mask])) if n > 1 else 0.0
        offdiag_pr_std = float(np.std(pr_matrix[mask])) if n > 1 else 0.0
        offdiag_pr_min = float(np.min(pr_matrix[mask])) if n > 1 else 0.0

        episode_ids = [ep["episode_id"] for ep in episodes]

        results[task_name] = {
            "num_episodes": n,
            "episode_ids": episode_ids,
            "state_recall_matrix": sr_matrix.tolist(),
            "path_recall_matrix": pr_matrix.tolist(),
            "diag_state_recall": diag_sr,
            "diag_path_recall": diag_pr,
            "offdiag_state_recall_mean": offdiag_sr,
            "offdiag_state_recall_std": offdiag_sr_std,
            "offdiag_path_recall_mean": offdiag_pr,
            "offdiag_path_recall_std": offdiag_pr_std,
            "offdiag_path_recall_min": offdiag_pr_min,
        }

    return results


def compute_cross_task_transfer(
    tasks: Dict[str, List[Dict]],
    variant: str = "pipeline_output",
) -> Dict:
    """Compute cross-task transfer using 3 deterministic representatives per task."""
    task_names = sorted(tasks.keys())
    if len(task_names) < 2:
        return {"error": "need at least 2 tasks"}

    # Select 3 deterministic representatives per task
    reps = {}
    for task_name in task_names:
        episodes = tasks[task_name]
        n = len(episodes)
        if n <= 3:
            indices = list(range(n))
        else:
            indices = [0, n // 3, 2 * n // 3]
        reps[task_name] = [episodes[i] for i in indices]

    # Preload lifted trajectories for all representatives
    rep_trajs = {}
    for task_name, rep_eps in reps.items():
        for ep in rep_eps:
            key = f"{task_name}/{ep['episode_id']}"
            path = ep["episode_dir"] / "lifted_trajectory.json"
            if path.exists():
                rep_trajs[key] = load_lifted_trajectory(path)

    # Build ordered list of all representatives
    all_reps = []
    task_labels = []
    for task_name in task_names:
        for ep in reps[task_name]:
            key = f"{task_name}/{ep['episode_id']}"
            all_reps.append((task_name, ep, key))
            task_labels.append(task_name)

    N = len(all_reps)
    sr_block = np.zeros((N, N))
    pr_block = np.zeros((N, N))

    for i in range(N):
        task_i, ep_i, key_i = all_reps[i]
        graph_dir = ep_i["episode_dir"] / variant
        for j in range(N):
            task_j, ep_j, key_j = all_reps[j]
            if key_j in rep_trajs:
                metrics = _eval_graph_on_trajectory(graph_dir, rep_trajs[key_j])
                sr_block[i, j] = metrics["state_recall"]
                pr_block[i, j] = metrics["path_recall"]

    # Compute within-task vs cross-task from block matrix
    within_sr, within_pr = [], []
    cross_sr, cross_pr = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if task_labels[i] == task_labels[j]:
                within_sr.append(sr_block[i, j])
                within_pr.append(pr_block[i, j])
            else:
                cross_sr.append(sr_block[i, j])
                cross_pr.append(pr_block[i, j])

    # Per-task cross-task means
    per_task_cross = {}
    for task_name in task_names:
        task_cross_pr = []
        for i in range(N):
            if task_labels[i] != task_name:
                continue
            for j in range(N):
                if task_labels[j] == task_name:
                    continue
                task_cross_pr.append(pr_block[i, j])
        per_task_cross[task_name] = {
            "cross_path_recall_mean": float(np.mean(task_cross_pr)) if task_cross_pr else 0.0,
            "cross_path_recall_std": float(np.std(task_cross_pr)) if task_cross_pr else 0.0,
        }

    return {
        "num_representatives": N,
        "task_labels": task_labels,
        "representative_ids": [r[2] for r in all_reps],
        "state_recall_block_matrix": sr_block.tolist(),
        "path_recall_block_matrix": pr_block.tolist(),
        "within_task_state_recall_mean": float(np.mean(within_sr)) if within_sr else 0.0,
        "within_task_path_recall_mean": float(np.mean(within_pr)) if within_pr else 0.0,
        "cross_task_state_recall_mean": float(np.mean(cross_sr)) if cross_sr else 0.0,
        "cross_task_path_recall_mean": float(np.mean(cross_pr)) if cross_pr else 0.0,
        "cross_task_state_recall_std": float(np.std(cross_sr)) if cross_sr else 0.0,
        "cross_task_path_recall_std": float(np.std(cross_pr)) if cross_pr else 0.0,
        "per_task_cross": per_task_cross,
    }


# ════════════════════════════════════════════════════════════════════
#  D. Task Separability & Statistical Tests
# ════════════════════════════════════════════════════════════════════

def compute_task_separability(
    within_transfer: Dict,
    cross_transfer: Dict,
) -> Dict:
    """Compute Δ_task and statistical significance."""
    task_within = {}
    for task_name, data in within_transfer.items():
        if data.get("skip"):
            continue
        task_within[task_name] = data["offdiag_path_recall_mean"]

    task_cross = {}
    for task_name, data in cross_transfer.get("per_task_cross", {}).items():
        task_cross[task_name] = data["cross_path_recall_mean"]

    common_tasks = sorted(set(task_within) & set(task_cross))
    if len(common_tasks) < 2:
        return {"error": "not enough tasks for statistical test"}

    within_vals = np.array([task_within[t] for t in common_tasks])
    cross_vals = np.array([task_cross[t] for t in common_tasks])
    gaps = within_vals - cross_vals

    # Macro means
    macro_within = float(np.mean(within_vals))
    macro_cross = float(np.mean(cross_vals))
    macro_gap = float(np.mean(gaps))
    gap_std = float(np.std(gaps))

    # Bootstrap 95% CI for the gap
    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.choice(len(gaps), size=len(gaps), replace=True)
        boot_gaps.append(float(np.mean(gaps[idx])))
    boot_gaps = sorted(boot_gaps)
    ci_low = boot_gaps[int(0.025 * n_boot)]
    ci_high = boot_gaps[int(0.975 * n_boot)]

    # Wilcoxon signed-rank test
    try:
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(within_vals, cross_vals, alternative="greater")
        wilcoxon_result = {"statistic": float(stat), "p_value": float(p_value)}
    except ImportError:
        wilcoxon_result = {"error": "scipy not installed"}
    except Exception as e:
        wilcoxon_result = {"error": str(e)}

    # Transfer ratio per task
    eps = 1e-6
    transfer_ratios = {}
    for t in common_tasks:
        transfer_ratios[t] = task_within[t] / (task_cross[t] + eps)

    return {
        "num_tasks": len(common_tasks),
        "tasks": common_tasks,
        "macro_within_offdiag_path_recall": macro_within,
        "macro_cross_task_path_recall": macro_cross,
        "macro_gap": macro_gap,
        "gap_std": gap_std,
        "bootstrap_95ci": [ci_low, ci_high],
        "wilcoxon_test": wilcoxon_result,
        "per_task_within": {t: float(task_within[t]) for t in common_tasks},
        "per_task_cross": {t: float(task_cross[t]) for t in common_tasks},
        "per_task_gap": {t: float(g) for t, g in zip(common_tasks, gaps)},
        "per_task_transfer_ratio": {t: float(v) for t, v in transfer_ratios.items()},
    }


# ════════════════════════════════════════════════════════════════════
#  E. Summary Table
# ════════════════════════════════════════════════════════════════════

def build_summary_table(
    tasks: Dict[str, List[Dict]],
    vocab: Dict,
    structure: Dict,
    within_full: Dict,
    within_pruned: Dict,
    within_raw: Dict,
    cross_task: Dict,
    separability: Dict,
) -> List[Dict]:
    """Build per-task summary rows + macro/micro aggregation."""
    rows = []
    for task_name in sorted(tasks.keys()):
        row = {"task": task_name, "num_episodes": len(tasks[task_name])}

        v = vocab.get(task_name, {})
        row["pred_jaccard"] = v.get("predicate_jaccard_mean", None)
        row["action_jaccard"] = v.get("action_jaccard_mean", None)
        row["task_shared_frac"] = v.get("task_shared_fraction", None)

        s = structure.get(task_name, {})
        if not s.get("skip"):
            agg = s.get("aggregate", {})
            row["node_cv"] = agg.get("nodes", {}).get("cv", None)
            row["edge_cv"] = agg.get("edges", {}).get("cv", None)
            row["scc_cv"] = agg.get("scc_count", {}).get("cv", None)

        wf = within_full.get(task_name, {})
        if not wf.get("skip"):
            row["diag_sr"] = wf.get("diag_state_recall")
            row["diag_pr"] = wf.get("diag_path_recall")
            row["offdiag_sr_full"] = wf.get("offdiag_state_recall_mean")
            row["offdiag_pr_full"] = wf.get("offdiag_path_recall_mean")

        wp = within_pruned.get(task_name, {})
        if not wp.get("skip"):
            row["offdiag_sr_pruned"] = wp.get("offdiag_state_recall_mean")
            row["offdiag_pr_pruned"] = wp.get("offdiag_path_recall_mean")

        wr = within_raw.get(task_name, {})
        if not wr.get("skip"):
            row["offdiag_sr_raw"] = wr.get("offdiag_state_recall_mean")
            row["offdiag_pr_raw"] = wr.get("offdiag_path_recall_mean")

        ct = cross_task.get("per_task_cross", {}).get(task_name, {})
        row["cross_task_pr"] = ct.get("cross_path_recall_mean")

        if row.get("offdiag_pr_full") is not None and row.get("cross_task_pr") is not None:
            row["gap_pr"] = row["offdiag_pr_full"] - row["cross_task_pr"]
        else:
            row["gap_pr"] = None

        rows.append(row)

    # Macro aggregation
    def _macro(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        return None

    macro = {k: _macro(k) for k in [
        "pred_jaccard", "diag_sr", "diag_pr",
        "offdiag_sr_full", "offdiag_pr_full",
        "offdiag_sr_pruned", "offdiag_pr_pruned",
        "offdiag_sr_raw", "offdiag_pr_raw",
        "cross_task_pr", "gap_pr",
        "node_cv", "scc_cv",
    ]}

    return {"per_task": rows, "macro": macro}


# ════════════════════════════════════════════════════════════════════
#  F. Figures
# ════════════════════════════════════════════════════════════════════

def generate_figures(
    tasks: Dict[str, List[Dict]],
    within_full: Dict,
    cross_task: Dict,
    within_pruned: Dict,
    within_raw: Dict,
    separability: Dict,
    vocab: Dict,
    output_dir: Path,
):
    """Generate all Experiment B figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed; skipping figures")
        return

    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        sns = None

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Within-task transfer heatmaps (2–3 representative tasks) ──
    sorted_tasks = sorted(
        [(t, d) for t, d in within_full.items() if not d.get("skip")],
        key=lambda x: x[1]["num_episodes"],
        reverse=True,
    )
    for task_name, data in sorted_tasks[:3]:
        n = data["num_episodes"]
        for metric, mat_key, label in [
            ("state_recall", "state_recall_matrix", "State Recall"),
            ("path_recall", "path_recall_matrix", "Path Recall"),
        ]:
            mat = np.array(data[mat_key])
            fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
            ax.set_title(f"{task_name}\n{label} Transfer Matrix", fontsize=10)
            ax.set_xlabel("Trajectory j")
            ax.set_ylabel("Graph from episode i")
            if n <= 15:
                eps = data["episode_ids"]
                short_ids = [e[:8] for e in eps]
                ax.set_xticks(range(n))
                ax.set_xticklabels(short_ids, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(n))
                ax.set_yticklabels(short_ids, fontsize=7)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            safe_name = task_name.replace("/", "_")[:50]
            fig.savefig(figures_dir / f"{safe_name}_{metric}_matrix.png", dpi=150)
            plt.close(fig)

    # ── Figure 2: Block matrix ──
    if "state_recall_block_matrix" in cross_task:
        for metric, mat_key, label in [
            ("state_recall", "state_recall_block_matrix", "State Recall"),
            ("path_recall", "path_recall_block_matrix", "Path Recall"),
        ]:
            mat = np.array(cross_task[mat_key])
            N = mat.shape[0]
            task_labels = cross_task["task_labels"]

            fig, ax = plt.subplots(figsize=(max(10, N * 0.15), max(8, N * 0.12)))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
            ax.set_title(f"Block Matrix — {label}\n(same-task blocks should be bright)", fontsize=11)

            # Draw block boundaries
            unique_tasks = []
            boundaries = [0]
            prev = task_labels[0]
            for i, t in enumerate(task_labels):
                if t != prev:
                    boundaries.append(i)
                    unique_tasks.append(prev)
                    prev = t
            boundaries.append(N)
            unique_tasks.append(prev)

            for b in boundaries[1:-1]:
                ax.axhline(b - 0.5, color="white", linewidth=1.5)
                ax.axvline(b - 0.5, color="white", linewidth=1.5)

            # Task labels at block centers
            centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(unique_tasks))]
            short_labels = [t[:15] for t in unique_tasks]
            ax.set_yticks(centers)
            ax.set_yticklabels(short_labels, fontsize=6)
            ax.set_xticks(centers)
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6)

            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            fig.savefig(figures_dir / f"block_matrix_{metric}.png", dpi=150)
            plt.close(fig)

    # ── Figure 3: Task-level bar plot with per-task p-values ──
    sep = separability
    if "tasks" in sep:
        from scipy.stats import wilcoxon as _wilcoxon, mannwhitneyu as _mannwhitneyu

        task_list = sep["tasks"]
        within_vals = [sep["per_task_within"][t] for t in task_list]
        cross_vals = [sep["per_task_cross"][t] for t in task_list]
        gaps = [w - c for w, c in zip(within_vals, cross_vals)]
        order = sorted(range(len(task_list)), key=lambda i: gaps[i], reverse=True)

        # Global Wilcoxon test
        try:
            _, p_global = _wilcoxon(within_vals, cross_vals)
        except Exception:
            p_global = float("nan")

        # Per-task p-values: compare within-task off-diag PR distribution
        # vs cross-task PR distribution using Mann-Whitney U
        per_task_p = {}
        # Get cross-task block matrix data
        block_pr = np.array(cross_task["path_recall_block_matrix"]) if "path_recall_block_matrix" in cross_task else None
        block_labels = cross_task.get("task_labels", [])

        for t in task_list:
            # Within-task off-diagonal values
            wf = within_full.get(t, {})
            if wf.get("skip"):
                per_task_p[t] = float("nan")
                continue
            pr_mat = np.array(wf["path_recall_matrix"])
            n = pr_mat.shape[0]
            mask = ~np.eye(n, dtype=bool)
            within_samples = pr_mat[mask].tolist()

            # Cross-task values for this task from block matrix
            cross_samples = []
            if block_pr is not None:
                for i in range(len(block_labels)):
                    if block_labels[i] != t:
                        continue
                    for j in range(len(block_labels)):
                        if block_labels[j] == t:
                            continue
                        cross_samples.append(block_pr[i, j])

            if len(within_samples) >= 2 and len(cross_samples) >= 2:
                try:
                    _, p = _mannwhitneyu(within_samples, cross_samples, alternative="two-sided")
                except Exception:
                    p = float("nan")
            else:
                p = float("nan")
            per_task_p[t] = p

        fig, ax = plt.subplots(figsize=(max(12, len(task_list) * 0.9), 7))
        x = np.arange(len(task_list))
        w = 0.35
        ax.bar(x - w / 2, [within_vals[i] for i in order], w, label="Within-task off-diag", color="#2196F3")
        ax.bar(x + w / 2, [cross_vals[i] for i in order], w, label="Cross-task", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels([task_list[i][:20] for i in order], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Path Recall")
        ax.set_title("Within-Task vs Cross-Task Path Recall (sorted by gap)")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.25)

        # Annotate per-task p-values above each bar pair
        for xi, idx in enumerate(order):
            t = task_list[idx]
            p = per_task_p.get(t, float("nan"))
            top = max(within_vals[idx], cross_vals[idx])
            if np.isfinite(p):
                if p < 0.001:
                    p_str = "***"
                elif p < 0.01:
                    p_str = "**"
                elif p < 0.05:
                    p_str = "*"
                else:
                    p_str = "n.s."
                label = f"p={p:.3f}\n{p_str}"
            else:
                label = "N/A"
            ax.text(xi, top + 0.02, label, ha="center", va="bottom", fontsize=6.5)

        # Global annotation
        if np.isfinite(p_global):
            g_text = f"Global Wilcoxon (H₀: equal): p = {p_global:.3f}"
        else:
            g_text = "Global Wilcoxon: N/A"
        ax.text(0.5, 1.06, g_text, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

        plt.tight_layout()
        fig.savefig(figures_dir / "within_vs_cross_task.png", dpi=150)
        plt.close(fig)

    # ── Figure 4: Summary boxplot with CLD ──
    box_data = []
    box_labels = []
    for task_name, data in within_full.items():
        if data.get("skip"):
            continue
        box_data.append(data["diag_path_recall"])
        box_labels.append("Self")
    for task_name, data in within_full.items():
        if data.get("skip"):
            continue
        box_data.append(data["offdiag_path_recall_mean"])
        box_labels.append("Within-task\noff-diag")
    if "per_task_cross" in cross_task:
        for task_name in within_full:
            if within_full[task_name].get("skip"):
                continue
            ct = cross_task["per_task_cross"].get(task_name, {})
            box_data.append(ct.get("cross_path_recall_mean", 0))
            box_labels.append("Cross-task")

    if box_data:
        unique_labels = ["Self", "Within-task\noff-diag", "Cross-task"]
        grouped = {l: [] for l in unique_labels}
        for v, l in zip(box_data, box_labels):
            grouped[l].append(v)
        present = [l for l in unique_labels if grouped[l]]

        # Pairwise Wilcoxon tests for CLD
        from scipy.stats import wilcoxon as _wilcoxon
        alpha = 0.05
        n_groups = len(present)
        pairwise_p = np.ones((n_groups, n_groups))
        min_len = min(len(grouped[l]) for l in present)
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                a = grouped[present[i]][:min_len]
                b = grouped[present[j]][:min_len]
                try:
                    _, p = _wilcoxon(a, b)
                except Exception:
                    p = 1.0
                pairwise_p[i, j] = p
                pairwise_p[j, i] = p

        # Build compact letter display
        letters = [set() for _ in range(n_groups)]
        current_letter = ord("A")
        for i in range(n_groups):
            same_group = [i]
            for j in range(n_groups):
                if i != j and pairwise_p[i, j] >= alpha:
                    same_group.append(j)
            letter = None
            for m in same_group:
                for existing in letters[m]:
                    if all(existing in letters[k] for k in same_group if letters[k]):
                        letter = existing
                        break
                if letter:
                    break
            if letter is None:
                letter = chr(current_letter)
                current_letter += 1
            for m in same_group:
                letters[m].add(letter)
        cld = ["".join(sorted(s)) for s in letters]

        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(
            [grouped[l] for l in present],
            labels=present,
            patch_artist=True,
        )
        colors = ["#4CAF50", "#2196F3", "#FF9800"]
        for patch, color in zip(bp["boxes"], colors[:len(present)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Path Recall")
        ax.set_title("Path Recall Distribution Across Tasks")
        ax.set_ylim(0, 1.2)

        # Add CLD letters above each box
        for i, (label, letter_str) in enumerate(zip(present, cld)):
            upper = max(grouped[label])
            ax.text(i + 1, upper + 0.04, letter_str, ha="center", va="bottom",
                    fontsize=14, fontweight="bold")

        # Add p-value annotation table below title
        p_lines = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                p = pairwise_p[i, j]
                short_i = present[i].replace("\n", " ")
                short_j = present[j].replace("\n", " ")
                p_lines.append(f"{short_i} vs {short_j}: p={p:.3f}")
        ax.text(0.02, 0.98, "\n".join(p_lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

        plt.tight_layout()
        fig.savefig(figures_dir / "summary_boxplot.png", dpi=150)
        plt.close(fig)

    # ── Figure 6: Three-level ablation with CLD ──
    ablation_data = {"raw": [], "pruned": [], "full": []}
    for task_name in sorted(within_full.keys()):
        if within_full[task_name].get("skip"):
            continue
        ablation_data["full"].append(within_full[task_name]["offdiag_path_recall_mean"])
        if task_name in within_pruned and not within_pruned[task_name].get("skip"):
            ablation_data["pruned"].append(within_pruned[task_name]["offdiag_path_recall_mean"])
        else:
            ablation_data["pruned"].append(0)
        if task_name in within_raw and not within_raw[task_name].get("skip"):
            ablation_data["raw"].append(within_raw[task_name]["offdiag_path_recall_mean"])
        else:
            ablation_data["raw"].append(0)

    if ablation_data["full"]:
        from scipy.stats import wilcoxon as _wilcoxon

        conditions = ["Raw\n(Step 0-2)", "Pruned\n(+ mutex)", "Full\n(pipeline)"]
        keys = ["raw", "pruned", "full"]
        means = [np.mean(ablation_data[k]) for k in keys]
        stds = [np.std(ablation_data[k]) for k in keys]

        # Pairwise Wilcoxon tests
        alpha = 0.05
        n_groups = 3
        pairwise_p = np.ones((n_groups, n_groups))
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                try:
                    _, p = _wilcoxon(ablation_data[keys[i]], ablation_data[keys[j]])
                except Exception:
                    p = 1.0
                pairwise_p[i, j] = p
                pairwise_p[j, i] = p

        # Build compact letter display
        letters = [set() for _ in range(n_groups)]
        current_letter = ord("A")
        for i in range(n_groups):
            same_group = [i]
            for j in range(n_groups):
                if i != j and pairwise_p[i, j] >= alpha:
                    same_group.append(j)
            letter = None
            for m in same_group:
                for existing in letters[m]:
                    if all(existing in letters[k] for k in same_group if letters[k]):
                        letter = existing
                        break
                if letter:
                    break
            if letter is None:
                letter = chr(current_letter)
                current_letter += 1
            for m in same_group:
                letters[m].add(letter)
        cld = ["".join(sorted(s)) for s in letters]

        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(3)
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#f44336", "#ff9800", "#4caf50"], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("Off-Diagonal Path Recall (macro)")
        ax.set_title("Three-Level Ablation: Graph Construction")
        ax.set_ylim(0, 1.2)

        # Add mean values and CLD letters
        for i, (bar, mean, letter_str) in enumerate(zip(bars, means, cld)):
            top = bar.get_height() + stds[i]
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.07,
                    letter_str, ha="center", va="bottom", fontsize=14, fontweight="bold")

        # Add p-value annotation
        p_lines = []
        short_names = ["Raw", "Pruned", "Full"]
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                p = pairwise_p[i, j]
                p_lines.append(f"{short_names[i]} vs {short_names[j]}: p={p:.3f}")
        ax.text(0.02, 0.98, "\n".join(p_lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

        plt.tight_layout()
        fig.savefig(figures_dir / "three_level_ablation.png", dpi=150)
        plt.close(fig)

    logger.info("Saved figures to %s", figures_dir)


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: within-task graph consistency analysis.",
    )
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Base experiment output directory (from batch_evaluate_all.py).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    experiment_dir = Path(args.experiment_dir)
    analysis_dir = experiment_dir / "consistency_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover episodes ──
    logger.info("Discovering completed episodes ...")
    tasks = discover_episodes(experiment_dir)
    logger.info("Found %d tasks with %d total episodes",
                len(tasks), sum(len(eps) for eps in tasks.values()))
    for task_name, eps in sorted(tasks.items()):
        logger.info("  %s: %d episodes", task_name, len(eps))

    # ── A. Vocabulary Consistency ──
    logger.info("=" * 60)
    logger.info("A. Vocabulary Consistency")
    vocab = analyze_vocabulary(tasks)
    _save_json(analysis_dir / "vocabulary_consistency.json", vocab)
    for t, v in sorted(vocab.items()):
        if not v.get("skip"):
            logger.info("  %s: pred_jaccard=%.3f action_jaccard=%.3f shared=%.1f%%",
                        t, v["predicate_jaccard_mean"], v["action_jaccard_mean"],
                        v["task_shared_fraction"] * 100)

    # ── B. Structural Consistency ──
    logger.info("=" * 60)
    logger.info("B. Structural Consistency")
    structure = analyze_structure(tasks)
    _save_json(analysis_dir / "structural_consistency.json", structure)
    for t, s in sorted(structure.items()):
        if not s.get("skip"):
            agg = s["aggregate"]
            logger.info("  %s: nodes=%.1f±%.1f edges=%.1f±%.1f SCC=%.1f±%.1f",
                        t,
                        agg["nodes"]["mean"], agg["nodes"]["std"],
                        agg["edges"]["mean"], agg["edges"]["std"],
                        agg["scc_count"]["mean"], agg["scc_count"]["std"])

    # ── C. Within-task transfer (3 variants) ──
    logger.info("=" * 60)
    logger.info("C. Within-task transfer — Full Pipeline")
    within_full = compute_within_task_transfer(tasks, "pipeline_output")
    _save_json(analysis_dir / "within_task_transfer_full.json", within_full)

    logger.info("C. Within-task transfer — Baseline Pruned")
    within_pruned = compute_within_task_transfer(tasks, "baseline_pruned_output")
    _save_json(analysis_dir / "within_task_transfer_pruned.json", within_pruned)

    logger.info("C. Within-task transfer — Baseline Raw")
    within_raw = compute_within_task_transfer(tasks, "baseline_raw_output")
    _save_json(analysis_dir / "within_task_transfer_raw.json", within_raw)

    for t, d in sorted(within_full.items()):
        if not d.get("skip"):
            p = within_pruned.get(t, {})
            r = within_raw.get(t, {})
            logger.info(
                "  %s: diag_PR=%.3f offdiag_PR=%.3f (full) / %.3f (pruned) / %.3f (raw)",
                t, d["diag_path_recall"], d["offdiag_path_recall_mean"],
                p.get("offdiag_path_recall_mean", 0),
                r.get("offdiag_path_recall_mean", 0),
            )

    # ── D. Cross-task transfer ──
    logger.info("=" * 60)
    logger.info("D. Cross-task transfer")
    cross_task = compute_cross_task_transfer(tasks, "pipeline_output")
    _save_json(analysis_dir / "cross_task_transfer.json", cross_task)
    if "error" not in cross_task:
        logger.info(
            "  Within-task SR=%.3f PR=%.3f | Cross-task SR=%.3f PR=%.3f",
            cross_task["within_task_state_recall_mean"],
            cross_task["within_task_path_recall_mean"],
            cross_task["cross_task_state_recall_mean"],
            cross_task["cross_task_path_recall_mean"],
        )

    # ── E. Task separability ──
    logger.info("=" * 60)
    logger.info("E. Task Separability")
    separability = compute_task_separability(within_full, cross_task)
    _save_json(analysis_dir / "task_separability.json", separability)
    if "error" not in separability:
        logger.info(
            "  Macro gap (PR): %.3f [95%% CI: %.3f, %.3f]",
            separability["macro_gap"],
            separability["bootstrap_95ci"][0],
            separability["bootstrap_95ci"][1],
        )
        if "p_value" in separability.get("wilcoxon_test", {}):
            logger.info("  Wilcoxon p-value: %.4f", separability["wilcoxon_test"]["p_value"])

    # ── F. Summary table ──
    logger.info("=" * 60)
    logger.info("F. Building summary table")
    summary = build_summary_table(
        tasks, vocab, structure,
        within_full, within_pruned, within_raw,
        cross_task, separability,
    )
    _save_json(analysis_dir / "summary_table.json", summary)

    # CSV export
    import csv
    csv_path = analysis_dir / "summary_table.csv"
    if summary["per_task"]:
        fieldnames = list(summary["per_task"][0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary["per_task"]:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
        logger.info("  Saved CSV: %s", csv_path)

    # ── G. Figures ──
    logger.info("=" * 60)
    logger.info("G. Generating figures")
    generate_figures(
        tasks, within_full, cross_task,
        within_pruned, within_raw,
        separability, vocab, analysis_dir,
    )

    # ── Final summary ──
    logger.info("=" * 60)
    logger.info("EXPERIMENT B — SUMMARY")
    logger.info("=" * 60)
    if "error" not in separability:
        logger.info("Primary metric (Within-task off-diag Path Recall): %.3f",
                     separability["macro_within_offdiag_path_recall"])
        logger.info("Negative control (Cross-task Path Recall):         %.3f",
                     separability["macro_cross_task_path_recall"])
        logger.info("Task separability gap Δ_task:                      %.3f",
                     separability["macro_gap"])
        logger.info("95%% Bootstrap CI:                                  [%.3f, %.3f]",
                     separability["bootstrap_95ci"][0], separability["bootstrap_95ci"][1])
    macro = summary.get("macro", {})
    if macro.get("offdiag_pr_full") and macro.get("offdiag_pr_pruned") and macro.get("offdiag_pr_raw"):
        logger.info("Ablation (off-diag PR): raw=%.3f → pruned=%.3f → full=%.3f",
                     macro["offdiag_pr_raw"]["mean"],
                     macro["offdiag_pr_pruned"]["mean"],
                     macro["offdiag_pr_full"]["mean"])
    logger.info("All results saved to: %s", analysis_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
