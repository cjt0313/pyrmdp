#!/usr/bin/env python3
"""
evaluate_video_trajectories.py — Evaluate empirical trajectories against
the pyrmdp abstract graph.

Takes RoboMIND HDF5 trajectories (or a directory of them), queries a VLM
to estimate per-frame logical states, filters noise via Exactly-One mutex
groups, lifts to abstract predicates, and computes recall metrics against
the synthesized abstract graph.

The abstract graph can be supplied via --pipeline-dir (pre-built) or built
automatically from the trajectory's first frame and task description.

Usage
-----
  # Auto-build the abstract graph, use all cameras (default):
  python scripts/evaluate_video_trajectories.py \\
      --trajectory pyrmdp/test_failure_data/trajectory.hdf5 \\
      --target-fps 2 --source-fps 10 \\
      --max-workers 4 \\
      --output evaluation_results.json -v

  # Use a pre-built pipeline directory with specific cameras:
  python scripts/evaluate_video_trajectories.py \\
      --pipeline-dir pyrmdp/test_data/7/output_batch \\
      --trajectory pyrmdp/test_failure_data/trajectory.hdf5 \\
      --cameras camera_front camera_wrist \\
      --target-fps 2 --source-fps 10 \\
      --max-workers 4 \\
      --output evaluation_results.json -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Ensure the package is importable when run from the repo root ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from pyrmdp.offline_validate.trajectory_lifter import (
    lift_trajectory,
    lift_state,
)
from pyrmdp.offline_validate.graph_evaluator import (
    load_abstract_graph,
    graph_predicate_vocabulary,
    evaluate,
)

from run_pipeline import run_pipeline

logger = logging.getLogger("pyrmdp.eval")


# ════════════════════════════════════════════════════════════════════
#  Pipeline data helpers
# ════════════════════════════════════════════════════════════════════

def _load_predicate_signatures(pipeline_dir: Path) -> List[str]:
    """Parse predicate signatures from the generated PDDL domain."""
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

    logger.info("Loaded %d predicate signatures from %s", len(sigs), pddl_path.name)
    return sigs


def _find_trajectories(path: Path) -> List[Path]:
    """Resolve trajectory path(s) — single file or directory of HDF5s."""
    if path.is_file() and path.suffix == ".hdf5":
        return [path]
    if path.is_dir():
        found = sorted(path.glob("**/*.hdf5"))
        if not found:
            raise FileNotFoundError(f"No .hdf5 files found in {path}")
        return found
    raise FileNotFoundError(f"Trajectory path not found: {path}")


def _parse_pred_params(pred_sigs: List[str]) -> Dict[str, str]:
    """Build a mapping from bare predicate name to its PDDL parameter string.

    Input sig example: ``'holding ?r - robot ?m - movable'``
    Output: ``{'holding': '?r, ?m'}``
    """
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
                lines.append((f"\u2203: {pred}({params})", changed))
            else:
                changed = pred not in prev_false and i > 0
                lines.append((f"\u00ac\u2203: {pred}({params})", changed))

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

    logger.info("  Saved %d annotated frames to %s", len(frames), output_dir)


# ════════════════════════════════════════════════════════════════════
#  Single trajectory evaluation
# ════════════════════════════════════════════════════════════════════

def evaluate_single(
    traj_path: Path,
    pipeline_dir: Path,
    pred_sigs: List[str],
    vlm_fn,
    abstract_graph,
    mutex_rules: List[Dict],
    exactly_one_groups,
    *,
    cameras: List[str] | None = None,
    target_fps: float,
    source_fps: float,
    max_workers: int,
    embodiment: str | None,
    frame_output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline on one trajectory."""

    t0 = time.time()

    # ── Step 1: Load & subsample frames (all cameras) ──
    logger.info("── Loading trajectory: %s", traj_path.name)
    if cameras is None:
        cameras = list_cameras(traj_path)
    logger.info("  Using cameras: %s", cameras)

    traj = load_trajectory_multicam(
        traj_path,
        cameras=cameras,
        target_fps=target_fps,
        source_fps=source_fps,
        embodiment=embodiment,
    )
    logger.info(
        "  Task: '%s' | %d frames sampled from %d total (%d cameras)",
        traj.language_instruction, len(traj.frames), traj.total_frames,
        len(cameras),
    )

    # ── Step 2: Bootstrap grounding (all camera views of frame 0) ──
    graph_vocab = graph_predicate_vocabulary(abstract_graph)
    active_sigs = [s for s in pred_sigs if s.split()[0] in graph_vocab]
    logger.info(
        "── Bootstrap: discovering grounded predicates "
        "(%d/%d sigs active in graph) …",
        len(active_sigs), len(pred_sigs),
    )
    grounded_preds = discover_grounding(
        vlm_fn, traj.frames[0], active_sigs, traj.language_instruction,
    )
    logger.info("  Grounded predicates: %s", grounded_preds)

    # ── Step 3: VLM state estimation (parallel, multi-camera) ──
    logger.info("── Estimating frame states via VLM (%d frames × %d cameras) …",
                len(traj.frames), len(cameras))
    raw_states = estimate_trajectory(
        vlm_fn, traj.frames, grounded_preds, max_workers=max_workers,
    )

    # ── Step 3.5: Save annotated frames (front camera view) ──
    if frame_output_dir is not None:
        traj_out = frame_output_dir / traj_path.stem
        front_idx = cameras.index("camera_front") if "camera_front" in cameras else 0
        front_frames = [views[front_idx] for views in traj.frames]
        _save_annotated_frames(front_frames, raw_states, traj_out, pred_sigs=pred_sigs)

    # ── Step 4: Mutex-bounded temporal smoothing ──
    logger.info("── Applying mutex filter …")
    keyframes = filter_and_collapse(raw_states, exactly_one_groups, mutex_rules)

    if not keyframes:
        logger.warning("  All frames filtered — no valid keyframes remain")
        return {
            "trajectory": str(traj_path),
            "language_instruction": traj.language_instruction,
            "total_frames": traj.total_frames,
            "sampled_frames": len(traj.frames),
            "valid_frames": 0,
            "keyframes": 0,
            "state_recall": 0.0,
            "path_recall": 0.0,
            "super_coverage": 0.0,
            "elapsed_s": time.time() - t0,
        }

    # ── Step 5: Lift to abstract signatures ──
    logger.info("── Lifting %d keyframes to abstract signatures …", len(keyframes))
    lifted = lift_trajectory(keyframes)

    for i, (tp, fp) in enumerate(lifted):
        logger.debug("  K%d: true=%s  false=%s", i, sorted(tp), sorted(fp))

    # ── Step 6: Graph evaluation ──
    logger.info("── Evaluating against abstract graph …")
    result = evaluate(lifted, abstract_graph)

    for i, bs in enumerate(result.belief_states):
        logger.debug("  K%d belief state: %s (%d nodes)", i, bs, len(bs))

    elapsed = time.time() - t0

    return {
        "trajectory": str(traj_path),
        "language_instruction": traj.language_instruction,
        "cameras": cameras,
        "total_frames": traj.total_frames,
        "sampled_frames": len(traj.frames),
        "valid_frames": len(keyframes),
        "keyframes": len(keyframes),
        "grounded_predicates": grounded_preds,
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
        "elapsed_s": elapsed,
    }


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video trajectories against pyrmdp abstract graphs.",
    )
    parser.add_argument(
        "--pipeline-dir", "-p", default=None,
        help="Path to a pre-built pipeline output directory. If omitted, the "
             "abstract graph is synthesized from the trajectory's first frame.",
    )
    parser.add_argument(
        "--synthesis-output-dir", default="./pipeline_output",
        help="Where to store synthesized pipeline output when --pipeline-dir "
             "is not provided (default: ./pipeline_output).",
    )
    parser.add_argument(
        "--trajectory", "-t", required=True,
        help="Path to an HDF5 trajectory file or directory of HDF5 files.",
    )
    parser.add_argument(
        "--cameras", nargs="*", default=None,
        help="Camera keys to use (default: all cameras in the HDF5 file).",
    )
    parser.add_argument(
        "--target-fps", type=float, default=2.0,
        help="Target frame extraction rate in Hz (default: 2.0).",
    )
    parser.add_argument(
        "--source-fps", type=float, default=10.0,
        help="Source recording rate in Hz (default: 10.0).",
    )
    parser.add_argument(
        "--embodiment", default=None,
        help="Embodiment tag for BGR correction (e.g., h5_franka_3rgb).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Max parallel VLM workers (default: 4).",
    )
    parser.add_argument(
        "--output", "-o", default="evaluation_results.json",
        help="Output JSON path (default: evaluation_results.json).",
    )
    parser.add_argument(
        "--frame-output-dir", default=None,
        help="Directory to save annotated frame images with predicate states.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline_dir = Path(args.pipeline_dir) if args.pipeline_dir else None
    traj_path = Path(args.trajectory)

    # ── Discover trajectories ──
    trajectories = _find_trajectories(traj_path)
    logger.info("Found %d trajectory file(s) to evaluate", len(trajectories))

    # ── Auto-synthesize pipeline if --pipeline-dir not provided ──
    if pipeline_dir is None:
        logger.info("═" * 60)
        logger.info("No --pipeline-dir given; synthesizing abstract graph …")
        logger.info("═" * 60)

        first_traj = load_trajectory_multicam(
            trajectories[0],
            cameras=None,
            target_fps=args.target_fps,
            source_fps=args.source_fps,
            embodiment=args.embodiment,
        )

        frame_img = Image.fromarray(first_traj.frames[0][0].astype(np.uint8))
        synthesis_dir = Path(args.synthesis_output_dir)
        synthesis_dir.mkdir(parents=True, exist_ok=True)
        frame_path = synthesis_dir / "_first_frame.png"
        frame_img.save(str(frame_path))
        logger.info("  Saved first frame to %s", frame_path)

        cfg = PipelineConfig(
            output_dir=str(synthesis_dir),
            save_intermediates=True,
        )
        run_pipeline(
            image_paths=[str(frame_path)],
            task_descriptions=[first_traj.language_instruction],
            config=cfg,
        )
        pipeline_dir = synthesis_dir
        logger.info("Pipeline synthesis complete → %s", pipeline_dir)

    # ── Load pipeline data (once for all trajectories) ──
    logger.info("═" * 60)
    logger.info("Loading pipeline data from: %s", pipeline_dir)
    logger.info("═" * 60)

    pred_sigs = _load_predicate_signatures(pipeline_dir)
    mutex_rules = load_mutex_rules(pipeline_dir)
    exactly_one_groups = extract_exactly_one_groups(mutex_rules)
    abstract_graph = load_abstract_graph(pipeline_dir)

    logger.info("Building VLM function from llm.yaml …")
    llm_config = load_config()
    vlm_fn = build_vlm_fn(llm_config)

    # ── Evaluate each trajectory ──
    all_results: List[Dict[str, Any]] = []
    for tp in trajectories:
        result = evaluate_single(
            tp, pipeline_dir, pred_sigs, vlm_fn, abstract_graph,
            mutex_rules, exactly_one_groups,
            cameras=args.cameras,
            target_fps=args.target_fps,
            source_fps=args.source_fps,
            max_workers=args.max_workers,
            embodiment=args.embodiment,
            frame_output_dir=Path(args.frame_output_dir) if args.frame_output_dir else None,
        )
        all_results.append(result)

        logger.info("")
        logger.info("─── Results: %s ───", tp.name)
        logger.info("  State Recall:    %.2f (%d/%d)",
                     result["state_recall"],
                     result["matched_keyframes"],
                     result["total_keyframes"])
        logger.info("  Path Recall:     %.2f (%d/%d)",
                     result["path_recall"],
                     result["matched_transitions"],
                     result["total_transitions"])
        logger.info("  Super-Coverage:  %.2f (%d graph / %d empirical)",
                     result["super_coverage"],
                     result["total_graph_nodes"],
                     result["unique_empirical_states"])
        logger.info("  Time:            %.1fs", result["elapsed_s"])
        logger.info("")

    # ── Aggregate metrics ──
    n = len(all_results)
    agg = {
        "num_trajectories": n,
        "mean_state_recall": sum(r["state_recall"] for r in all_results) / n if n else 0,
        "mean_path_recall": sum(r["path_recall"] for r in all_results) / n if n else 0,
        "mean_super_coverage": sum(r["super_coverage"] for r in all_results) / n if n else 0,
        "per_trajectory": all_results,
    }

    # ── Save ──
    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("═" * 60)
    logger.info("Saved evaluation results: %s", out_path)
    logger.info(
        "Aggregate:  State=%.2f  Path=%.2f  Coverage=%.2f",
        agg["mean_state_recall"],
        agg["mean_path_recall"],
        agg["mean_super_coverage"],
    )
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
