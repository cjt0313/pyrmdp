#!/usr/bin/env python3
"""
Experiment 1: Spectral Convergence Proof

Validates that the iterative hallucination loop terminates and stabilises
regardless of the per-iteration recovery-operator budget.

Sweep
-----
For each budget setting in {1, 2, 3, ∞} the full pipeline is executed
on every test case.  The per-iteration spectral distance, eigenvalue
spectrum, and abstract-state count are recorded.

Output
------
  <output_dir>/
    experiment1_results.json      ← aggregated data for plotting
    budget_1/case_1/              ← full pipeline outputs per (budget, case)
    budget_1/case_2/
    ...
    budget_inf/case_9/

Usage
-----
  # Full sweep (all 9 cases × 4 budgets):
  python scripts/run_experiment1_convergence.py

  # Quick test on one case:
  python scripts/run_experiment1_convergence.py --cases 6 --budgets 2

  # Custom output:
  python scripts/run_experiment1_convergence.py -o experiments/exp1_v2/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Ensure repo root is importable ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

TEST_DATA_DIR = _REPO_ROOT / "pyrmdp" / "test_data"

logger = logging.getLogger("experiment1")


# ════════════════════════════════════════════════════════════════════
#  Worker
# ════════════════════════════════════════════════════════════════════

def _run_one(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker: run the pipeline for one (budget, case) pair.

    Args tuple:
        (case_dir, budget_value, output_dir, max_loop_iters, num_policies)

    budget_value: int > 0 for a finite budget, or 0 for unlimited.
    """
    case_dir_str, budget_value, output_dir_str, max_loop_iters, num_policies = args_tuple
    case_dir = Path(case_dir_str)
    output_dir = Path(output_dir_str)
    case_id = case_dir.name
    budget_label = str(budget_value) if budget_value > 0 else "inf"

    result: Dict[str, Any] = {
        "case": case_id,
        "budget": budget_label,
        "budget_value": budget_value,
        "status": "unknown",
        "elapsed_s": 0.0,
        "output_dir": str(output_dir),
        "error": "",
    }

    t0 = time.time()

    try:
        # ── Logging per-process ──
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s [B{budget_label}-C{case_id}] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        log = logging.getLogger(f"exp1.B{budget_label}.C{case_id}")

        # ── Discover image + task files ──
        images = sorted(case_dir.glob("*.jpg")) + sorted(case_dir.glob("*.png"))
        txt_files = sorted(case_dir.glob("*.txt"))
        if not images or not txt_files:
            result["status"] = "skipped"
            result["error"] = "Missing image or task file"
            return result

        image_path = images[0]
        tasks = [
            line.strip()
            for line in txt_files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not tasks:
            result["status"] = "skipped"
            result["error"] = "Empty task file"
            return result

        log.info(
            "Starting: budget=%s, case=%s, image=%s, tasks=%d",
            budget_label, case_id, image_path.name, len(tasks),
        )

        # ── Import pipeline ──
        from pyrmdp.synthesis.config import PipelineConfig
        from run_pipeline import run_pipeline

        cfg = PipelineConfig(
            num_robot_policies=num_policies,
            max_recovery_per_iter=budget_value if budget_value > 0 else None,
            max_loop_iterations=max_loop_iters,
            output_dir=str(output_dir),
            save_intermediates=True,
            visualize=True,
        )

        ppddl = run_pipeline(
            image_paths=[str(image_path)],
            task_descriptions=tasks,
            config=cfg,
        )

        # ── Read back pipeline_summary.json ──
        summary_path = output_dir / "pipeline_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result["iterations"] = summary.get("iterations", 0)
            result["converged"] = summary.get("converged", False)
            result["spectral_distances"] = summary.get("spectral_distances", [])
            result["per_iteration"] = summary.get("per_iteration", [])
            result["total_actions"] = summary.get("total_actions", 0)
            result["epsilon"] = summary.get("epsilon", 0.05)
        else:
            result["iterations"] = 0
            result["converged"] = False

        result["status"] = "success"
        result["ppddl_lines"] = ppddl.count("\n") + 1
        log.info("✓ Done: %d iterations, converged=%s", result["iterations"], result["converged"])

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        logging.getLogger(f"exp1.B{budget_label}.C{case_id}").error(
            "✗ Failed:\n%s", traceback.format_exc()
        )

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def discover_cases(
    test_data_dir: Path,
    subset: Optional[List[str]] = None,
) -> List[Path]:
    """Find valid test case directories sorted numerically."""
    dirs = sorted(
        (d for d in test_data_dir.iterdir() if d.is_dir()),
        key=lambda p: (
            p.name.isdigit(),
            int(p.name) if p.name.isdigit() else 0,
            p.name,
        ),
    )
    if subset:
        dirs = [d for d in dirs if d.name in subset]
    return dirs


def print_budget_summary(budget_label: str, results: List[Dict]):
    """Print a summary table for one budget setting."""
    print(f"\n── Budget = {budget_label} ──")
    print(f"  {'Case':<6} {'Status':<8} {'Iters':>5} {'Conv':>5} {'Time':>7}  {'Δ_final':>10}")
    print("  " + "─" * 55)
    for r in sorted(results, key=lambda x: x["case"]):
        iters = r.get("iterations", "?")
        conv = "✓" if r.get("converged") else "✗"
        dists = r.get("spectral_distances", [])
        d_final = f"{dists[-1]:.6f}" if dists else "—"
        status = "✓" if r["status"] == "success" else "✗"
        print(
            f"  {r['case']:<6} {status} {r['status']:<6} {iters:>5} {conv:>5} "
            f"{r['elapsed_s']:>6.1f}s  {d_final:>10}"
        )


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Spectral Convergence Proof — budget sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--budgets", "-b",
        nargs="+", default=["1", "2", "3", "0"],
        help=(
            "Budget settings to sweep (0 = unlimited/∞). "
            "Default: 1 2 3 0"
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+", default=None,
        help="Run only these case IDs (e.g. --cases 1 4 6).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=3,
        help="Parallel workers per budget sweep (default: 3).",
    )
    parser.add_argument(
        "--max-loop-iterations",
        type=int, default=50,
        help="Hard cap on outer loop iterations (default: 50).",
    )
    parser.add_argument(
        "--num-policies", "-K",
        type=int, default=1,
        help="Robot policy variants per action (default: 1, fast).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(_REPO_ROOT / "experiments" / "exp1_convergence"),
        help="Root output directory for experiment results.",
    )
    parser.add_argument(
        "--test-data-dir",
        default=str(TEST_DATA_DIR),
        help=f"Path to test_data directory (default: {TEST_DATA_DIR}).",
    )

    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Discover cases ──
    cases = discover_cases(Path(args.test_data_dir), args.cases)
    if not cases:
        print(f"No test cases found in {args.test_data_dir}")
        sys.exit(1)

    print("═" * 70)
    print("  Experiment 1: Spectral Convergence Proof")
    print("═" * 70)
    print(f"  Budgets:  {[b if b > 0 else '∞' for b in budgets]}")
    print(f"  Cases:    {[c.name for c in cases]}")
    print(f"  Workers:  {args.workers}")
    print(f"  Max iter: {args.max_loop_iterations}")
    print(f"  Output:   {output_root}")
    print("═" * 70)

    all_results: Dict[str, Dict[str, Any]] = {}
    total_start = time.time()

    for budget in budgets:
        budget_label = str(budget) if budget > 0 else "inf"
        budget_dir = output_root / f"budget_{budget_label}"
        budget_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 70}")
        print(f"  Running budget = {budget_label}  ({len(cases)} cases)")
        print(f"{'─' * 70}")

        # Build worker args for this budget
        worker_args = [
            (
                str(case_dir),
                budget,
                str(budget_dir / f"case_{case_dir.name}"),
                args.max_loop_iterations,
                args.num_policies,
            )
            for case_dir in cases
        ]

        # Run in parallel
        sweep_start = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = pool.map(_run_one, worker_args)
        sweep_elapsed = time.time() - sweep_start

        # Collect results
        budget_results = {}
        for r in results:
            budget_results[r["case"]] = r
        all_results[budget_label] = budget_results

        print_budget_summary(budget_label, results)
        print(f"  Budget {budget_label} completed in {sweep_elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # ── Save aggregated results ──
    results_path = output_root / "experiment1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'═' * 70}")
    print(f"  Experiment 1 complete — {total_elapsed:.1f}s total")
    print(f"  Results: {results_path}")
    print(f"  Plot:    python scripts/plot_convergence.py {results_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
