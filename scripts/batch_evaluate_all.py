#!/usr/bin/env python3
"""
batch_evaluate_all.py — Batch evaluation of sampled trajectories with
three graph variants (full pipeline, baseline-pruned, baseline-raw).

Reads a sampled_tasks.json (from census_failure_data.py), runs the full
evaluation pipeline for each trajectory, and saves all intermediates for
downstream consistency analysis (analyze_consistency.py).

Usage
-----
  python scripts/batch_evaluate_all.py \
      --sampled-tasks experiment_output/sampled_tasks.json \
      --output-dir /inspire/qb-ilm/.../experiment_output \
      --target-fps 2 --source-fps 10 \
      --max-vlm-workers 4 \
      --resume -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
from PIL import Image
from tqdm import tqdm

from pyrmdp.synthesis.llm_config import load_config
from pyrmdp.synthesis.config import PipelineConfig
from pyrmdp.offline_validate.robomind.hdf5_loader import (
    load_trajectory_multicam,
    list_cameras,
)
from pyrmdp.offline_validate.vlm_state_estimator import (
    build_vlm_fn,
    discover_grounding,
    estimate_trajectory,
)
from pyrmdp.offline_validate.mutex_filter import (
    load_mutex_rules,
    extract_exactly_one_groups,
    filter_and_collapse,
)
from pyrmdp.offline_validate.trajectory_lifter import lift_trajectory
from pyrmdp.offline_validate.trajectory_lifter import lift_state
from pyrmdp.offline_validate.graph_evaluator import (
    load_abstract_graph,
    graph_predicate_vocabulary,
    evaluate,
)

from run_pipeline import run_pipeline

logger = logging.getLogger("pyrmdp.batch_eval")


def _load_predicate_signatures(pipeline_dir: Path) -> List[str]:
    """Parse predicate signatures from the generated PDDL domain."""
    import re
    pddl_path = pipeline_dir / "step0_domain_generated.pddl"
    if not pddl_path.exists():
        raise FileNotFoundError(f"Domain file not found: {pddl_path}")

    text = pddl_path.read_text(encoding="utf-8")
    start = text.find("(:predicates")
    if start < 0:
        raise ValueError("No :predicates block found in domain file")

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
    sigs = []
    for tok in re.findall(r"\(([^)]+)\)", block):
        tok = tok.strip()
        if not tok or tok[0] == ";":
            continue
        sigs.append(tok)
    return sigs


def _config_hash(cfg: PipelineConfig) -> str:
    """Short hash of config for metadata."""
    d = cfg.to_dict()
    d.pop("output_dir", None)
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _find_final_file(pipeline_dir: Path, pattern: str) -> Optional[str]:
    """Find the last iteration's file matching a glob pattern."""
    candidates = sorted(pipeline_dir.glob(pattern))
    if candidates:
        return candidates[-1].name
    return None


def _save_json(path: Path, data: Any) -> None:
    def _default(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    path.write_text(
        json.dumps(data, indent=2, default=_default, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_pred_params(pred_sigs: List[str]) -> Dict[str, str]:
    """Map bare predicate name to its PDDL parameter string."""
    result: Dict[str, str] = {}
    for sig in pred_sigs:
        parts = sig.split()
        name = parts[0]
        params = [p for p in parts[1:] if p.startswith("?")]
        result[name] = ", ".join(params) if params else "x"
    return result


def _save_annotated_frames(
    frames: List[np.ndarray],
    states: List[Dict[str, bool]],
    output_dir: Path,
    pred_sigs: List[str] | None = None,
    prefix: str = "frame",
) -> None:
    """Save each frame as a PNG with predicate states annotated top-left."""
    from PIL import ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    param_map = _parse_pred_params(pred_sigs) if pred_sigs else {}

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    prev_true, prev_false = frozenset(), frozenset()
    all_preds = sorted({p for s in states for p in lift_state(s)[0] | lift_state(s)[1]})

    for i, (frame, state) in enumerate(zip(frames, states)):
        img = Image.fromarray(frame.astype(np.uint8)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        true_preds, false_preds = lift_state(state)

        lines = []
        for pred in all_preds:
            params = param_map.get(pred, "x")
            if pred in true_preds:
                changed = pred not in prev_true and i > 0
                lines.append((f"∃: {pred}({params})", changed))
            else:
                changed = pred not in prev_false and i > 0
                lines.append((f"¬∃: {pred}({params})", changed))

        margin = 6
        y = margin
        max_w = 0
        line_positions = []
        for text, changed in lines:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            max_w = max(max_w, w)
            line_positions.append((text, y, w, h, changed))
            y += h + 2

        draw.rectangle(
            [0, 0, max_w + 2 * margin, y + margin],
            fill=(0, 0, 0, 180),
        )
        for text, ly, w, h, changed in line_positions:
            color = (255, 255, 0, 255) if changed else (255, 255, 255, 255)
            draw.text((margin, ly), text, font=font, fill=color)

        img = Image.alpha_composite(img, overlay).convert("RGB")
        img.save(str(output_dir / f"{prefix}_{i:04d}.png"))

        prev_true, prev_false = true_preds, false_preds


def evaluate_with_graph(
    lifted_trajectory: list,
    pipeline_dir: Path,
    pred_sigs: List[str],
) -> Dict[str, Any]:
    """Evaluate a lifted trajectory against a graph from a pipeline dir."""
    try:
        abstract_graph = load_abstract_graph(pipeline_dir)
    except FileNotFoundError:
        return {"error": "no abstract graph found", "state_recall": 0, "path_recall": 0}

    result = evaluate(lifted_trajectory, abstract_graph)
    return {
        "state_recall": result.state_recall,
        "path_recall": result.path_recall,
        "super_coverage": result.super_coverage,
        "matched_keyframes": result.matched_keyframes,
        "total_keyframes": result.total_keyframes,
        "matched_transitions": result.matched_transitions,
        "total_transitions": result.total_transitions,
        "unique_empirical_states": result.unique_empirical_states,
        "total_graph_nodes": result.total_graph_nodes,
        "belief_state_sizes": [len(bs) for bs in result.belief_states],
        "unmatched_states": result.unmatched_states,
    }


def process_single_trajectory(
    task_name: str,
    episode_info: Dict,
    output_base: Path,
    *,
    target_fps: float,
    source_fps: float,
    max_vlm_workers: int,
    embodiment: Optional[str],
    cameras: Optional[List[str]],
) -> Dict[str, Any]:
    """Process one trajectory: synthesize 3 graph variants, estimate states, evaluate."""

    episode_id = episode_info["episode_id"]
    hdf5_path = Path(episode_info["hdf5_path"])
    failure_reason = episode_info.get("failure_reason", "unknown")

    episode_dir = output_base / "per_trajectory" / task_name / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dir = episode_dir / "pipeline_output"
    pruned_dir = episode_dir / "baseline_pruned_output"
    raw_dir = episode_dir / "baseline_raw_output"

    t0 = time.time()

    # ── Load & subsample frames ──
    logger.info("Loading trajectory: %s", hdf5_path.name)
    if cameras is None:
        cameras_used = list_cameras(hdf5_path)
    else:
        cameras_used = cameras

    traj = load_trajectory_multicam(
        hdf5_path,
        cameras=cameras_used,
        target_fps=target_fps,
        source_fps=source_fps,
        embodiment=embodiment,
    )
    logger.info(
        "  Task: '%s' | %d frames sampled from %d total (%d cameras)",
        traj.language_instruction, len(traj.frames), traj.total_frames,
        len(cameras_used),
    )

    # ── Synthesize 3 graph variants ──
    frame_img = Image.fromarray(traj.frames[0][0].astype(np.uint8))

    # Full pipeline
    logger.info("  Synthesizing full pipeline graph ...")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    frame_path = pipeline_dir / "_first_frame.png"
    frame_img.save(str(frame_path))
    full_cfg = PipelineConfig(
        output_dir=str(pipeline_dir),
        save_intermediates=True,
        visualize=False,
    )
    try:
        run_pipeline(
            image_paths=[str(frame_path)],
            task_descriptions=[traj.language_instruction],
            config=full_cfg,
        )
    except Exception as e:
        logger.error("  Full pipeline failed: %s", e)
        raise

    # Baseline-pruned (Step 0-2 + mutex pruning)
    logger.info("  Synthesizing baseline-pruned graph ...")
    pruned_dir.mkdir(parents=True, exist_ok=True)
    frame_path_p = pruned_dir / "_first_frame.png"
    frame_img.save(str(frame_path_p))
    pruned_cfg = PipelineConfig(
        output_dir=str(pruned_dir),
        save_intermediates=True,
        max_loop_iterations=1,
        failure_prob=0.0,
        enable_mutex_pruning=True,
        enable_llm_feasibility=False,
        max_recovery_per_iter=0,
        visualize=False,
    )
    try:
        run_pipeline(
            image_paths=[str(frame_path_p)],
            task_descriptions=[traj.language_instruction],
            config=pruned_cfg,
        )
    except Exception as e:
        logger.error("  Baseline-pruned failed: %s", e)
        raise

    # Baseline-raw (Step 0-2 only, no mutex pruning)
    logger.info("  Synthesizing baseline-raw graph ...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame_path_r = raw_dir / "_first_frame.png"
    frame_img.save(str(frame_path_r))
    raw_cfg = PipelineConfig(
        output_dir=str(raw_dir),
        save_intermediates=True,
        max_loop_iterations=1,
        failure_prob=0.0,
        enable_mutex_pruning=False,
        enable_llm_feasibility=False,
        max_recovery_per_iter=0,
        visualize=False,
    )
    try:
        run_pipeline(
            image_paths=[str(frame_path_r)],
            task_descriptions=[traj.language_instruction],
            config=raw_cfg,
        )
    except Exception as e:
        logger.error("  Baseline-raw failed: %s", e)
        raise

    # ── VLM state estimation (shared across all variants) ──
    logger.info("  Loading pipeline data for VLM grounding ...")
    pred_sigs = _load_predicate_signatures(pipeline_dir)
    abstract_graph = load_abstract_graph(pipeline_dir)
    graph_vocab = graph_predicate_vocabulary(abstract_graph)
    active_sigs = [s for s in pred_sigs if s.split()[0] in graph_vocab]

    llm_config = load_config()
    vlm_fn = build_vlm_fn(llm_config)

    logger.info("  Discovering grounded predicates (%d active sigs) ...", len(active_sigs))
    grounded_preds = discover_grounding(
        vlm_fn, traj.frames[0], active_sigs, traj.language_instruction,
    )
    logger.info("  Grounded predicates: %s", grounded_preds)
    _save_json(episode_dir / "grounded_predicates.json", grounded_preds)

    logger.info("  Estimating frame states via VLM (%d frames) ...", len(traj.frames))
    raw_states = estimate_trajectory(
        vlm_fn, traj.frames, grounded_preds, max_workers=max_vlm_workers,
    )
    _save_json(episode_dir / "raw_states.json", raw_states)

    # ── Save annotated frames (front camera view) ──
    front_frames = [views[0] for views in traj.frames]
    _save_annotated_frames(
        front_frames, raw_states, episode_dir / "annotated_frames",
        pred_sigs=pred_sigs,
    )

    # ── Mutex filter (using full pipeline's mutex rules) ──
    logger.info("  Applying mutex filter ...")
    try:
        mutex_rules = load_mutex_rules(pipeline_dir)
        exactly_one_groups = extract_exactly_one_groups(mutex_rules)
    except FileNotFoundError:
        logger.warning("  No mutex rules found, skipping filter")
        mutex_rules = []
        exactly_one_groups = []

    keyframes = filter_and_collapse(raw_states, exactly_one_groups, mutex_rules)
    _save_json(episode_dir / "keyframes.json", keyframes)

    if not keyframes:
        logger.warning("  All frames filtered — no valid keyframes")
        empty_result = {
            "trajectory": str(hdf5_path),
            "language_instruction": traj.language_instruction,
            "cameras": cameras_used,
            "total_frames": traj.total_frames,
            "sampled_frames": len(traj.frames),
            "valid_frames": 0,
            "keyframes": 0,
            "state_recall": 0.0,
            "path_recall": 0.0,
            "super_coverage": 0.0,
            "elapsed_s": time.time() - t0,
        }
        _save_json(episode_dir / "evaluation_result.json", empty_result)
        _save_json(episode_dir / "lifted_trajectory.json", {"states": []})
        return empty_result

    # ── Lift to abstract signatures ──
    logger.info("  Lifting %d keyframes ...", len(keyframes))
    lifted = lift_trajectory(keyframes)
    lifted_serializable = [
        {"true": sorted(tp), "false": sorted(fp)}
        for tp, fp in lifted
    ]
    _save_json(episode_dir / "lifted_trajectory.json", {"states": lifted_serializable})

    # ── Evaluate against all 3 graph variants ──
    logger.info("  Evaluating against full pipeline graph ...")
    full_eval = evaluate_with_graph(lifted, pipeline_dir, pred_sigs)
    full_eval.update({
        "trajectory": str(hdf5_path),
        "language_instruction": traj.language_instruction,
        "cameras": cameras_used,
        "total_frames": traj.total_frames,
        "sampled_frames": len(traj.frames),
        "valid_frames": len(keyframes),
        "keyframes": len(keyframes),
        "grounded_predicates": grounded_preds,
        "elapsed_s": time.time() - t0,
    })
    _save_json(episode_dir / "evaluation_result.json", full_eval)

    logger.info("  Evaluating against baseline-pruned graph ...")
    pruned_eval = evaluate_with_graph(lifted, pruned_dir, pred_sigs)
    _save_json(episode_dir / "baseline_pruned_evaluation.json", pruned_eval)

    logger.info("  Evaluating against baseline-raw graph ...")
    raw_eval = evaluate_with_graph(lifted, raw_dir, pred_sigs)
    _save_json(episode_dir / "baseline_raw_evaluation.json", raw_eval)

    # ── Save metadata ──
    metadata = {
        "task": task_name,
        "episode_id": episode_id,
        "hdf5_path": str(hdf5_path),
        "failure_reason": failure_reason,
        "full_graph_path": _find_final_file(pipeline_dir, "iter*_step2_abstract_graph.graphml"),
        "full_domain_path": "step0_domain_generated.pddl" if (pipeline_dir / "step0_domain_generated.pddl").exists() else None,
        "full_mutex_path": _find_final_file(pipeline_dir, "iter*_step2_mutex_rules.json"),
        "baseline_pruned_graph_path": _find_final_file(pruned_dir, "iter*_step2_abstract_graph.graphml"),
        "baseline_pruned_domain_path": "step0_domain_generated.pddl" if (pruned_dir / "step0_domain_generated.pddl").exists() else None,
        "baseline_raw_graph_path": _find_final_file(raw_dir, "iter*_step2_abstract_graph.graphml"),
        "baseline_raw_domain_path": "step0_domain_generated.pddl" if (raw_dir / "step0_domain_generated.pddl").exists() else None,
        "lifted_trajectory_path": "lifted_trajectory.json",
        "keyframes_path": "keyframes.json",
        "grounded_predicates_path": "grounded_predicates.json",
        "raw_states_path": "raw_states.json",
        "config": {
            "seed": 42,
            "model": llm_config.model if hasattr(llm_config, "model") else "gpt-4o",
            "target_fps": target_fps,
            "source_fps": source_fps,
            "full_pipeline_config_hash": _config_hash(full_cfg),
            "pruned_config_hash": _config_hash(pruned_cfg),
            "raw_config_hash": _config_hash(raw_cfg),
        },
    }
    _save_json(episode_dir / "metadata.json", metadata)

    elapsed = time.time() - t0
    full_eval["elapsed_s"] = elapsed

    logger.info(
        "  Done: SR=%.2f PR=%.2f (full) | SR=%.2f PR=%.2f (pruned) | "
        "SR=%.2f PR=%.2f (raw) | %.1fs",
        full_eval.get("state_recall", 0), full_eval.get("path_recall", 0),
        pruned_eval.get("state_recall", 0), pruned_eval.get("path_recall", 0),
        raw_eval.get("state_recall", 0), raw_eval.get("path_recall", 0),
        elapsed,
    )

    return full_eval


def _worker_fn(args_tuple):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    task_name, ep, output_dir, kwargs = args_tuple
    key = f"{task_name}/{ep['episode_id']}"
    try:
        result = process_single_trajectory(task_name, ep, Path(output_dir), **kwargs)
        return {"key": key, "result": result, "error": None}
    except Exception as e:
        logging.getLogger("pyrmdp.batch_eval").error("FAILED: %s — %s", key, e)
        logging.getLogger("pyrmdp.batch_eval").debug(traceback.format_exc())
        return {"key": key, "result": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of sampled trajectories with 3 graph variants.",
    )
    parser.add_argument(
        "--sampled-tasks", required=True,
        help="Path to sampled_tasks.json from census_failure_data.py.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Base output directory for all results.",
    )
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--embodiment", default=None)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--max-vlm-workers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of trajectories to process in parallel.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed trajectories.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count trajectories without processing.")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Only process these specific tasks.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    sampled_path = Path(args.sampled_tasks)
    sampled = json.loads(sampled_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work list
    work_list = []
    for task_name, task_info in sorted(sampled["tasks"].items()):
        if args.tasks and task_name not in args.tasks:
            continue
        for ep in task_info["sampled_episodes"]:
            work_list.append((task_name, ep))

    logger.info("Total trajectories to process: %d", len(work_list))

    if args.dry_run:
        for task_name, ep in work_list:
            logger.info("  %s / %s", task_name, ep["episode_id"])
        return

    # Load checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    completed = set()
    if args.resume and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed = set(ckpt.get("completed", []))
        logger.info("Resuming: %d already completed", len(completed))

    # Process trajectories
    all_results = []
    failed = []

    # Filter out already-completed work
    pending_work = []
    for task_name, ep in work_list:
        key = f"{task_name}/{ep['episode_id']}"
        if key in completed:
            logger.info("SKIP (already done): %s", key)
            result_path = output_dir / "per_trajectory" / task_name / ep["episode_id"] / "evaluation_result.json"
            if result_path.exists():
                all_results.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            pending_work.append((task_name, ep))

    logger.info("Pending: %d trajectories, workers: %d", len(pending_work), args.workers)

    worker_kwargs = dict(
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        max_vlm_workers=args.max_vlm_workers,
        embodiment=args.embodiment,
        cameras=args.cameras,
    )

    if args.workers <= 1:
        # Sequential mode
        pbar = tqdm(pending_work, desc="Evaluating", unit="traj",
                    dynamic_ncols=True, ncols=120)
        for task_name, ep in pbar:
            key = f"{task_name}/{ep['episode_id']}"
            pbar.set_postfix_str(key[-60:], refresh=True)
            out = _worker_fn((task_name, ep, str(output_dir), worker_kwargs))
            if out["error"] is None:
                all_results.append(out["result"])
                completed.add(out["key"])
                sr = out["result"].get("state_recall", 0)
                pr = out["result"].get("path_recall", 0)
                pbar.set_postfix_str(
                    f"{key[-40:]} SR={sr:.2f} PR={pr:.2f}", refresh=True)
                _save_json(checkpoint_path, {
                    "completed": sorted(completed),
                    "total": len(work_list),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            else:
                failed.append({"key": out["key"], "error": out["error"]})
                pbar.set_postfix_str(f"{key[-40:]} FAILED", refresh=True)
        pbar.close()
    else:
        # Parallel mode
        mp.set_start_method("fork", force=True)
        submit_args = [
            (tn, ep, str(output_dir), worker_kwargs)
            for tn, ep in pending_work
        ]
        pbar = tqdm(total=len(pending_work), desc="Evaluating", unit="traj",
                    dynamic_ncols=True, ncols=120)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker_fn, a): a[0] + "/" + a[1]["episode_id"] for a in submit_args}
            for future in as_completed(futures):
                key_label = futures[future]
                out = future.result()
                if out["error"] is None:
                    all_results.append(out["result"])
                    completed.add(out["key"])
                    sr = out["result"].get("state_recall", 0)
                    pr = out["result"].get("path_recall", 0)
                    _save_json(checkpoint_path, {
                        "completed": sorted(completed),
                        "total": len(work_list),
                        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    pbar.set_postfix_str(
                        f"{key_label[-40:]} SR={sr:.2f} PR={pr:.2f}", refresh=True)
                else:
                    failed.append({"key": out["key"], "error": out["error"]})
                    pbar.set_postfix_str(f"{key_label[-40:]} FAILED", refresh=True)
                pbar.update(1)
        pbar.close()

    # ── Aggregate results ──
    n = len(all_results)
    if n > 0:
        agg = {
            "num_trajectories": n,
            "num_failed": len(failed),
            "mean_state_recall": sum(r.get("state_recall", 0) for r in all_results) / n,
            "mean_path_recall": sum(r.get("path_recall", 0) for r in all_results) / n,
            "mean_super_coverage": sum(r.get("super_coverage", 0) for r in all_results) / n,
            "per_trajectory": all_results,
            "failed": failed,
        }
    else:
        agg = {"num_trajectories": 0, "num_failed": len(failed), "failed": failed}

    _save_json(output_dir / "all_results.json", agg)

    logger.info("=" * 60)
    logger.info("Batch evaluation complete: %d succeeded, %d failed", n, len(failed))
    if n > 0:
        logger.info(
            "Aggregate (full pipeline): SR=%.3f  PR=%.3f  SC=%.3f",
            agg["mean_state_recall"], agg["mean_path_recall"], agg["mean_super_coverage"],
        )
    if failed:
        logger.warning("Failed trajectories:")
        for f in failed:
            logger.warning("  %s: %s", f["key"], f["error"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
