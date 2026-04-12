#!/usr/bin/env python3
"""
Run the full pyrmdp pipeline on every subfolder in test_data/ using
multiprocessing.

Each subfolder N/ is expected to contain:
    N.jpg   — scene image
    N.txt   — task descriptions (one per line)

Results are written to N/output_batch/ with intermediates + evolution.html.

Usage
-----
    # Run all 9 test cases with 3 workers (default):
    python scripts/run_all_testdata.py

    # Custom parallelism & policies:
    python scripts/run_all_testdata.py --workers 4 --num-policies 5

    # Run a subset:
    python scripts/run_all_testdata.py --cases 1 3 7

    # Dry-run (just print what would be executed):
    python scripts/run_all_testdata.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Ensure repo root is importable ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

TEST_DATA_DIR = _REPO_ROOT / "pyrmdp" / "test_data"


# ════════════════════════════════════════════════════════════════════
#  Worker function (runs in a child process)
# ════════════════════════════════════════════════════════════════════

def _run_one_case(args: Tuple) -> Dict:
    """
    Execute the pipeline for a single test case.

    Parameters are packed into a tuple for use with Pool.map():
        (case_dir, num_policies, output_name, verbose,
         max_recovery_per_iter, max_loop_iterations, epsilon)

    Returns a result dict with status, timing, and any error info.
    """
    case_dir, num_policies, output_name, verbose, max_recovery_per_iter, max_loop_iterations, epsilon = args
    case_dir = Path(case_dir)
    case_id = case_dir.name

    result = {
        "case": case_id,
        "status": "unknown",
        "elapsed_s": 0.0,
        "output_dir": "",
        "error": "",
    }

    t0 = time.time()

    try:
        # ── Configure logging per-process ──
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format=f"%(asctime)s [case-{case_id}] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            force=True,  # Reset handlers in child process
        )
        logger = logging.getLogger(f"batch.case-{case_id}")

        # ── Discover files ──
        images = sorted(case_dir.glob("*.jpg")) + sorted(case_dir.glob("*.png"))
        txt_files = sorted(case_dir.glob("*.txt"))

        if not images:
            result["status"] = "skipped"
            result["error"] = "No image file found"
            return result
        if not txt_files:
            result["status"] = "skipped"
            result["error"] = "No task file found"
            return result

        image_path = images[0]
        task_file = txt_files[0]
        tasks = [
            line.strip()
            for line in task_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not tasks:
            result["status"] = "skipped"
            result["error"] = "Task file is empty"
            return result

        out_dir = case_dir / output_name
        result["output_dir"] = str(out_dir)

        logger.info(
            "Starting: image=%s, tasks=%d, policies=%d → %s",
            image_path.name, len(tasks), num_policies, out_dir,
        )

        # ── Import pipeline (deferred to child process) ──
        from pyrmdp.synthesis.config import PipelineConfig
        from run_pipeline import run_pipeline

        cfg_kwargs = dict(
            num_robot_policies=num_policies,
            output_dir=str(out_dir),
            save_intermediates=True,
            visualize=True,
            max_recovery_per_iter=max_recovery_per_iter,
            max_loop_iterations=max_loop_iterations,
        )
        if epsilon is not None:
            cfg_kwargs["epsilon"] = epsilon
        cfg = PipelineConfig(**cfg_kwargs)

        ppddl = run_pipeline(
            image_paths=[str(image_path)],
            task_descriptions=tasks,
            config=cfg,
        )

        result["status"] = "success"
        result["ppddl_lines"] = ppddl.count("\n") + 1
        logger.info("✓ Done: %d lines of PPDDL", result["ppddl_lines"])

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        logging.getLogger(f"batch.case-{case_id}").error(
            "✗ Failed:\n%s", traceback.format_exc()
        )

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


# ════════════════════════════════════════════════════════════════════
#  Orchestrator
# ════════════════════════════════════════════════════════════════════

def discover_cases(
    test_data_dir: Path,
    subset: Optional[List[str]] = None,
) -> List[Path]:
    """Find all valid test case directories (sorted numerically)."""
    dirs = sorted(
        (d for d in test_data_dir.iterdir() if d.is_dir()),
        key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name),
    )
    if subset:
        dirs = [d for d in dirs if d.name in subset]
    return dirs


def print_summary(results: List[Dict], total_time: float):
    """Pretty-print the batch results table."""
    print("\n" + "═" * 72)
    print("  BATCH RESULTS")
    print("═" * 72)
    print(f"  {'Case':<6} {'Status':<10} {'Time':>8}  {'Details'}")
    print("  " + "─" * 66)

    success = failed = skipped = 0
    for r in sorted(results, key=lambda x: x["case"]):
        status = r["status"]
        time_str = f"{r['elapsed_s']:.1f}s"

        if status == "success":
            mark = "✓"
            detail = f"{r.get('ppddl_lines', '?')} lines → {r['output_dir']}"
            success += 1
        elif status == "failed":
            mark = "✗"
            detail = r["error"]
            failed += 1
        else:
            mark = "–"
            detail = r["error"]
            skipped += 1

        print(f"  {r['case']:<6} {mark} {status:<8} {time_str:>8}  {detail}")

    print("  " + "─" * 66)
    print(
        f"  Total: {len(results)} cases "
        f"({success} success, {failed} failed, {skipped} skipped) "
        f"in {total_time:.1f}s"
    )
    print("═" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Run the pyrmdp pipeline on all test_data subfolders in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=3,
        help="Number of parallel workers (default: 3).",
    )
    parser.add_argument(
        "--num-policies", "-K",
        type=int, default=3,
        help="Number of robot policy variants per action (default: 3).",
    )
    parser.add_argument(
        "--output-name",
        default="output_batch",
        help="Name of the output sub-directory inside each case folder (default: output_batch).",
    )
    parser.add_argument(
        "--cases",
        nargs="+", default=None,
        help="Run only these case IDs (e.g. --cases 1 3 7).",
    )
    parser.add_argument(
        "--test-data-dir",
        default=str(TEST_DATA_DIR),
        help=f"Path to test_data directory (default: {TEST_DATA_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running.",
    )
    parser.add_argument(
        "--max-recovery-per-iter",
        type=int, default=None,
        help="Max recovery operators per iteration (budget cap for Step 5). Default: unlimited.",
    )
    parser.add_argument(
        "--epsilon",
        type=float, default=None,
        help="Wasserstein spectral-distance convergence threshold (default: use config default 0.1).",
    )
    parser.add_argument(
        "--max-loop-iterations",
        type=int, default=10,
        help="Max outer loop iterations (default: 10).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging in worker processes.",
    )

    args = parser.parse_args()

    # ── Discover cases ──
    test_dir = Path(args.test_data_dir)
    cases = discover_cases(test_dir, args.cases)

    if not cases:
        print(f"No test cases found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(cases)} test case(s): {[c.name for c in cases]}")
    print(f"Workers: {args.workers}, Policies: {args.num_policies}")
    print(f"Output: <case>/{args.output_name}/")

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for c in cases:
            imgs = sorted(c.glob("*.jpg")) + sorted(c.glob("*.png"))
            txts = sorted(c.glob("*.txt"))
            img_name = imgs[0].name if imgs else "???"
            txt_name = txts[0].name if txts else "???"
            tasks = []
            if txts:
                tasks = [l.strip() for l in txts[0].read_text().splitlines() if l.strip()]
            print(
                f"  Case {c.name}: {img_name} + {txt_name} "
                f"({len(tasks)} task{'s' if len(tasks) != 1 else ''}) "
                f"→ {c.name}/{args.output_name}/"
            )
        return

    # ── Build worker args ──
    worker_args = [
        (str(c), args.num_policies, args.output_name, args.verbose,
         args.max_recovery_per_iter, args.max_loop_iterations, args.epsilon)
        for c in cases
    ]

    # ── Run ──
    t0 = time.time()
    print(f"\nLaunching {len(worker_args)} pipeline(s) across {args.workers} worker(s)…\n")

    # Use spawn to avoid fork-safety issues with CUDA / OpenAI clients
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.map(_run_one_case, worker_args)

    total_time = time.time() - t0
    print_summary(results, total_time)

    # Exit with non-zero if any failed
    if any(r["status"] == "failed" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
