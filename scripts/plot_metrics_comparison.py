#!/usr/bin/env python3
"""
Plot 3-metric spectral convergence comparison.

Reads pipeline_summary.json files and plots cosine, L2, and Wasserstein
distances on the same axes for each case. If eigenvalues are stored but
a metric was not computed at runtime (e.g. Wasserstein for older runs),
it is recomputed retroactively from the saved eigenvalue arrays.

Usage
-----
  # Single case (existing data):
  python scripts/plot_metrics_comparison.py \
      pyrmdp/test_data/6/output_exp1_cos/pipeline_summary.json

  # Multiple cases from experiment 1:
  python scripts/plot_metrics_comparison.py \
      experiments/exp1_all/budget_1/case_*/pipeline_summary.json

  # Entire experiment directory (auto-discovers summaries):
  python scripts/plot_metrics_comparison.py experiments/exp1_all/

  # Custom output path:
  python scripts/plot_metrics_comparison.py experiments/exp1_all/ \
      -o figures/convergence_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.spatial.distance import cosine as _cosine_dist
from scipy.stats import wasserstein_distance as _wasserstein_1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


# ════════════════════════════════════════════════════════════════════
#  Metric computation (retroactive from eigenvalue arrays)
# ════════════════════════════════════════════════════════════════════

def _compute_metrics(
    eigenvalues_list: List[Optional[List[float]]],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute cosine, L2, and Wasserstein distances between consecutive
    eigenvalue arrays.

    Returns three lists of length (n-1) — one value per consecutive pair.
    """
    cosines, l2s, wassersteins = [], [], []

    for i in range(1, len(eigenvalues_list)):
        prev_ev = eigenvalues_list[i - 1]
        curr_ev = eigenvalues_list[i]

        if prev_ev is None or curr_ev is None:
            cosines.append(float("nan"))
            l2s.append(float("nan"))
            wassersteins.append(float("nan"))
            continue

        prev = np.sort(np.abs(prev_ev))[::-1]
        curr = np.sort(np.abs(curr_ev))[::-1]

        # ── Cosine (zero-padded) ──
        max_len = max(len(curr), len(prev))
        c = np.zeros(max_len)
        p = np.zeros(max_len)
        c[: len(curr)] = curr
        p[: len(prev)] = prev
        if np.linalg.norm(c) < 1e-12 or np.linalg.norm(p) < 1e-12:
            d_cos = 0.0
        else:
            d_cos = float(_cosine_dist(c, p))
        cosines.append(d_cos)

        # ── L2 (zero-padded) ──
        l2s.append(float(np.linalg.norm(c - p)))

        # ── Wasserstein (no padding) ──
        wassersteins.append(float(_wasserstein_1d(curr, prev)))

    return cosines, l2s, wassersteins


# ════════════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════════════

def _load_case(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Load and validate a pipeline_summary.json file."""
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠ Skipping {summary_path}: {e}", file=sys.stderr)
        return None

    pi = data.get("per_iteration", [])
    if len(pi) < 2:
        return None  # need ≥2 iterations for a distance

    # Extract eigenvalues
    eigenvalues = [it.get("eigenvalues") for it in pi]

    # Use stored metrics if available, otherwise recompute
    stored_cos = data.get("spectral_distances", [])
    stored_l2 = data.get("spectral_distances_l2", [])
    stored_w = data.get("spectral_distances_wasserstein", [])

    if stored_cos and stored_l2 and stored_w:
        cosines, l2s, wassersteins = stored_cos, stored_l2, stored_w
    else:
        # Recompute from eigenvalues
        cosines, l2s, wassersteins = _compute_metrics(eigenvalues)

    # Derive a label from the path
    # e.g. "budget_1/case_6" or "output_exp1_cos"
    parts = summary_path.parts
    label_parts = []
    for part in parts:
        if part.startswith("case_") or part.startswith("budget_"):
            label_parts.append(part)
        elif part.startswith("output_exp"):
            label_parts.append(part)
    if not label_parts:
        # Fallback: use parent dir name
        label_parts = [summary_path.parent.name]
    label = "/".join(label_parts)

    # Extract case id
    case_id = None
    for part in parts:
        if part.startswith("case_"):
            case_id = part.replace("case_", "")
            break
    # Try from test_data path
    if case_id is None:
        for j, part in enumerate(parts):
            if part == "test_data" and j + 1 < len(parts):
                case_id = parts[j + 1]
                break
    if case_id is None:
        case_id = summary_path.parent.name

    n_iters = len(pi)
    states = [it.get("states", 0) for it in pi]
    converged = data.get("converged", False)
    epsilon = data.get("epsilon", 0.02)

    return {
        "path": str(summary_path),
        "label": label,
        "case_id": case_id,
        "n_iters": n_iters,
        "converged": converged,
        "epsilon": epsilon,
        "states": states,
        "cosines": cosines,
        "l2s": l2s,
        "wassersteins": wassersteins,
    }


def discover_summaries(paths: List[str]) -> List[Path]:
    """Resolve paths — expand directories, glob patterns, etc."""
    result = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_file() and p.name == "pipeline_summary.json":
            result.append(p)
        elif p.is_dir():
            result.extend(sorted(p.rglob("pipeline_summary.json")))
        else:
            # Try glob from cwd
            from glob import glob
            result.extend(Path(m) for m in sorted(glob(p_str)))
    # Deduplicate
    seen = set()
    unique = []
    for r in result:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            unique.append(r)
    return unique


# ════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════

def plot_single_case(ax: plt.Axes, case: Dict[str, Any], show_epsilon: bool = True):
    """Plot 3 metrics on one Axes for a single case."""
    n = len(case["cosines"])
    iters = np.arange(2, 2 + n)  # iteration numbers (distance starts at iter 2)

    ax.plot(iters, case["cosines"], "o-", color="#2196F3", linewidth=2.0,
            markersize=5, label="Cosine distance", zorder=3)
    ax.plot(iters, case["wassersteins"], "s-", color="#4CAF50", linewidth=2.0,
            markersize=5, label="Wasserstein (EMD)", zorder=3)
    ax.plot(iters, case["l2s"], "^--", color="#FF9800", linewidth=1.5,
            markersize=5, alpha=0.7, label="L2 norm", zorder=2)

    if show_epsilon:
        ax.axhline(y=case["epsilon"], color="#F44336", linestyle=":",
                    linewidth=1.5, alpha=0.8, label=f"ε = {case['epsilon']}")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Spectral Distance", fontsize=11)
    ax.set_title(f"Case {case['case_id']}  ({case['n_iters']} iters, "
                 f"{'✓ converged' if case['converged'] else '✗ not converged'})",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def plot_all_cases(cases: List[Dict[str, Any]], output_path: str):
    """Create a figure with one subplot per case, all with 3-metric comparison."""
    n_cases = len(cases)
    if n_cases == 0:
        print("No cases with ≥2 iterations found.", file=sys.stderr)
        return

    # Layout: arrange subplots in a grid
    if n_cases == 1:
        nrows, ncols = 1, 1
        fig_w, fig_h = 8, 5
    elif n_cases <= 3:
        nrows, ncols = 1, n_cases
        fig_w, fig_h = 6 * n_cases, 5
    elif n_cases <= 6:
        nrows, ncols = 2, (n_cases + 1) // 2
        fig_w, fig_h = 6 * ncols, 5 * nrows
    else:
        ncols = 3
        nrows = (n_cases + ncols - 1) // ncols
        fig_w, fig_h = 6 * ncols, 4.5 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              squeeze=False)

    for idx, case in enumerate(cases):
        row, col = divmod(idx, ncols)
        plot_single_case(axes[row][col], case)

    # Hide unused axes
    for idx in range(n_cases, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Spectral Convergence — 3-Metric Comparison (budget = 1)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {out}")
    plt.close(fig)

    # Also save a combined overlay if multiple cases
    if n_cases > 1:
        _plot_overlay(cases, output_path)


_ZERO_FLOOR = 1e-10  # stand-in for 0.0 in log-space


def _smooth_gp(x: np.ndarray, y: np.ndarray, x_fine: np.ndarray) -> np.ndarray:
    """Fit a GP to (x, y) and return predictions at x_fine."""
    X = x.reshape(-1, 1).astype(float)
    kernel = ConstantKernel(1.0) * Matern(length_scale=3.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                  alpha=1e-6, normalize_y=True)
    gp.fit(X, y)
    return gp.predict(x_fine.reshape(-1, 1))


_METRICS = {
    "Cosine":      {"key": "cosines",      "color": "#2196F3", "marker": "o"},
    "Wasserstein": {"key": "wassersteins",  "color": "#4CAF50", "marker": "s"},
    "L2":          {"key": "l2s",           "color": "#FF9800", "marker": "^"},
}


def _plot_overlay(cases: List[Dict[str, Any]], base_output_path: str):
    """
    Overlay plot: one GP-smoothed mean curve per metric (log scale).

    Only non-converged cases are included.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── Keep only cases that did NOT converge (ran full iterations) ──
    full_cases = [c for c in cases if not c.get("converged", False)]
    if not full_cases:
        full_cases = cases  # fallback: use all if every case converged

    for name, cfg in _METRICS.items():
        # ── Build per-case arrays ──
        per_case: List[np.ndarray] = []
        for case in full_cases:
            v = np.array(case[cfg["key"]], dtype=float)
            per_case.append(v)
        max_n = max(len(v) for v in per_case)

        # ── Matrix (n_cases × max_n) ──
        n_cases = len(per_case)
        matrix = np.full((n_cases, max_n), np.nan)
        for ci, v in enumerate(per_case):
            matrix[ci, : len(v)] = v

        iters = np.arange(2, 2 + max_n, dtype=float)
        emp_mean = np.nanmean(matrix, axis=0)

        # ── GP-smooth mean ──
        n_fine = 200
        x_fine = np.linspace(iters[0], iters[-1], n_fine)
        mean_line = _smooth_gp(iters, emp_mean, x_fine)

        # ── Scatter raw points ──
        for v in per_case:
            n = len(v)
            it = np.arange(2, 2 + n)
            ax.scatter(it, v, color=cfg["color"], marker=cfg["marker"],
                       s=18, alpha=0.30, zorder=2)

        # ── GP-smoothed mean ──
        ax.plot(x_fine, mean_line, color=cfg["color"], linewidth=2.5,
                label=name, zorder=4)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Spectral Distance", fontsize=12)
    ax.set_title("Spectral Convergence — GP Regression per Metric",
                 fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    fig.tight_layout()

    out = Path(base_output_path).with_name(
        Path(base_output_path).stem + "_overlay.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved overlay: {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot 3-metric spectral convergence comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths", nargs="+",
        help="pipeline_summary.json files or directories to search",
    )
    parser.add_argument(
        "-o", "--output", default="figures/metrics_comparison.png",
        help="Output figure path (default: figures/metrics_comparison.png)",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Skip the combined overlay plot",
    )

    args = parser.parse_args()

    # Discover files
    summaries = discover_summaries(args.paths)
    if not summaries:
        print(f"No pipeline_summary.json found in: {args.paths}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(summaries)} summary file(s):")
    for s in summaries:
        print(f"  {s}")

    # Load and filter
    cases = []
    for sp in summaries:
        c = _load_case(sp)
        if c is not None:
            cases.append(c)
            print(f"  ✓ Case {c['case_id']}: {c['n_iters']} iters, "
                  f"converged={c['converged']}")
        else:
            print(f"  ⚠ Skipped (< 2 iterations): {sp}")

    if not cases:
        print("No cases with sufficient data for plotting.", file=sys.stderr)
        sys.exit(1)

    # Sort by case_id
    cases.sort(key=lambda c: (c["case_id"].isdigit(),
                               int(c["case_id"]) if c["case_id"].isdigit() else 0,
                               c["case_id"]))

    # Plot
    plot_all_cases(cases, args.output)

    # Summary table
    print(f"\n{'Case':>6}  {'Iters':>5}  {'Conv':>5}  {'Δ_cos':>10}  {'Δ_L2':>10}  {'Δ_W':>10}")
    print("-" * 58)
    for c in cases:
        d_cos = f"{c['cosines'][-1]:.6f}" if c['cosines'] else "—"
        d_l2 = f"{c['l2s'][-1]:.4f}" if c['l2s'] else "—"
        d_w = f"{c['wassersteins'][-1]:.6f}" if c['wassersteins'] else "—"
        conv = "✓" if c["converged"] else "✗"
        print(f"{c['case_id']:>6}  {c['n_iters']:>5}  {conv:>5}  {d_cos:>10}  {d_l2:>10}  {d_w:>10}")


if __name__ == "__main__":
    main()
