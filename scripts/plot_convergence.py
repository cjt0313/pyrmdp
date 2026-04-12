#!/usr/bin/env python3
"""
Plot spectral convergence curves from Experiment 1 data.

Produces publication-quality figures:

  Figure A — **Budget sweep on a single case**
    Δ_spectral (Y, log) vs. iteration (X), one line per budget setting,
    ε threshold as horizontal dashed line.

  Figure B — **All cases with a single budget**
    Δ_spectral vs. iteration for all cases (one curve each), showing
    they all collapse below ε.

  Figure C — **Abstract state-space growth**
    Number of abstract states (Y) vs. iteration (X), colored by budget.

Usage
-----
  # Plot all figures from experiment results:
  python scripts/plot_convergence.py experiments/exp1_convergence/experiment1_results.json

  # Specify which case / budget to highlight:
  python scripts/plot_convergence.py results.json --fig-a-case 1 --fig-b-budget 2

  # Custom output directory:
  python scripts/plot_convergence.py results.json -o figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ════════════════════════════════════════════════════════════════════
#  Publication style
# ════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    # Fonts — serif for camera-ready, fallback to DejaVu
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    # Sizes (NeurIPS single-column = 5.5 in)
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    # Axes
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Color palette — distinguishable, colorblind-friendly
BUDGET_STYLE = {
    "1":   {"color": "#d62728", "marker": "s", "label": r"$B = 1$"},
    "2":   {"color": "#ff7f0e", "marker": "^", "label": r"$B = 2$"},
    "3":   {"color": "#2ca02c", "marker": "D", "label": r"$B = 3$"},
    "inf": {"color": "#1f77b4", "marker": "o", "label": r"$B = \infty$"},
}

# 9 distinct colors for case lines
CASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


# ════════════════════════════════════════════════════════════════════
#  Data helpers
# ════════════════════════════════════════════════════════════════════

def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load experiment1_results.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_convergence_curve(case_data: Dict[str, Any]):
    """
    Extract (iterations, spectral_distances, state_counts) from a
    single case result.

    Returns
    -------
    iters : list[int]
        Iteration numbers (only those with a spectral distance).
    deltas : list[float]
        Corresponding spectral distances.
    all_iters : list[int]
        All iteration numbers.
    state_counts : list[int]
        Abstract state count per iteration.
    scc_counts : list[int]
        SCC count per iteration.
    """
    per_iter = case_data.get("per_iteration", [])
    iters, deltas = [], []
    all_iters, state_counts, scc_counts = [], [], []

    for it in per_iter:
        all_iters.append(it["iteration"])
        state_counts.append(it.get("states", 0))
        scc_counts.append(it.get("sccs", 0))
        sd = it.get("spectral_distance")
        if sd is not None:
            iters.append(it["iteration"])
            deltas.append(sd)

    return iters, deltas, all_iters, state_counts, scc_counts


# ════════════════════════════════════════════════════════════════════
#  Figure A: Budget sweep on one case
# ════════════════════════════════════════════════════════════════════

def plot_figure_a(
    data: Dict[str, Dict[str, Any]],
    case_id: str,
    epsilon: float,
    output_dir: Path,
):
    """
    Δ_spectral vs. iteration for a single case, all budget settings.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    budget_order = ["1", "2", "3", "inf"]
    has_data = False

    for budget in budget_order:
        if budget not in data:
            continue
        case_data = data[budget].get(case_id)
        if not case_data or case_data.get("status") != "success":
            continue

        iters, deltas, _, _, _ = extract_convergence_curve(case_data)
        if not iters:
            continue

        has_data = True
        style = BUDGET_STYLE[budget]
        ax.plot(
            iters, deltas,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )

    if not has_data:
        print(f"  ⚠ No convergence data found for case {case_id}")
        plt.close(fig)
        return

    # ε threshold line
    ax.axhline(
        y=epsilon, color="gray", linestyle="--", linewidth=0.8,
        label=rf"$\varepsilon = {epsilon}$", zorder=1,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta_{\cos}$")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(
        frameon=True, fancybox=False, edgecolor="0.7",
        loc="upper right",
    )
    ax.set_title(f"Case {case_id}: Budget Sweep")

    for fmt in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"fig_a_case{case_id}_budget_sweep.{fmt}")
    plt.close(fig)
    print(f"  ✓ Figure A saved (case {case_id})")


# ════════════════════════════════════════════════════════════════════
#  Figure B: All cases, single budget
# ════════════════════════════════════════════════════════════════════

def plot_figure_b(
    data: Dict[str, Dict[str, Any]],
    budget: str,
    epsilon: float,
    output_dir: Path,
):
    """
    Δ_spectral vs. iteration for all cases with one budget setting.
    """
    if budget not in data:
        print(f"  ⚠ Budget '{budget}' not found in results")
        return

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    budget_data = data[budget]

    case_ids = sorted(budget_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    has_data = False

    for i, cid in enumerate(case_ids):
        case_data = budget_data[cid]
        if case_data.get("status") != "success":
            continue

        iters, deltas, _, _, _ = extract_convergence_curve(case_data)
        if not iters:
            continue

        has_data = True
        color = CASE_COLORS[i % len(CASE_COLORS)]
        ax.plot(
            iters, deltas, "o-",
            color=color,
            label=f"Case {cid}",
            markersize=3.5,
            markeredgewidth=0.4,
            markeredgecolor="white",
            zorder=3,
        )

    if not has_data:
        print(f"  ⚠ No convergence data for budget {budget}")
        plt.close(fig)
        return

    ax.axhline(
        y=epsilon, color="gray", linestyle="--", linewidth=0.8,
        label=rf"$\varepsilon = {epsilon}$", zorder=1,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta_{\cos}$")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    budget_label = BUDGET_STYLE.get(budget, {}).get("label", f"B={budget}")
    ax.set_title(f"All Cases — {budget_label}")

    ax.legend(
        frameon=True, fancybox=False, edgecolor="0.7",
        loc="upper right", ncol=2, fontsize=7.5,
    )

    for fmt in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"fig_b_budget{budget}_all_cases.{fmt}")
    plt.close(fig)
    print(f"  ✓ Figure B saved (budget {budget})")


# ════════════════════════════════════════════════════════════════════
#  Figure C: Abstract state-space growth
# ════════════════════════════════════════════════════════════════════

def plot_figure_c(
    data: Dict[str, Dict[str, Any]],
    case_id: str,
    output_dir: Path,
):
    """
    Abstract state count (Y) vs. iteration (X), one line per budget.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)

    budget_order = ["1", "2", "3", "inf"]
    has_data = False

    for budget in budget_order:
        if budget not in data:
            continue
        case_data = data[budget].get(case_id)
        if not case_data or case_data.get("status") != "success":
            continue

        _, _, all_iters, state_counts, scc_counts = extract_convergence_curve(case_data)
        if not all_iters:
            continue

        has_data = True
        style = BUDGET_STYLE[budget]

        ax1.plot(
            all_iters, state_counts,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        ax2.plot(
            all_iters, scc_counts,
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    if not has_data:
        print(f"  ⚠ No data for case {case_id}")
        plt.close(fig)
        return

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Abstract States")
    ax1.set_title(f"Case {case_id}: State-Space Growth")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=8)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("SCCs")
    ax2.set_title(f"Case {case_id}: SCC Count")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=8)

    fig.tight_layout(w_pad=2.0)

    for fmt in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"fig_c_case{case_id}_state_growth.{fmt}")
    plt.close(fig)
    print(f"  ✓ Figure C saved (case {case_id})")


# ════════════════════════════════════════════════════════════════════
#  Summary table (LaTeX-ready)
# ════════════════════════════════════════════════════════════════════

def print_summary_table(data: Dict[str, Dict[str, Any]], epsilon: float):
    """Print a summary table of convergence stats for the paper."""
    print("\n" + "═" * 75)
    print("  Convergence Summary Table")
    print("═" * 75)
    print(
        f"  {'Budget':<8} {'Case':<6} {'Iters':>5} {'Conv':>5} "
        f"{'Δ_final':>10} {'States':>7} {'Actions':>8}"
    )
    print("  " + "─" * 68)

    for budget in ["1", "2", "3", "inf"]:
        if budget not in data:
            continue
        for cid in sorted(data[budget].keys(), key=lambda x: int(x) if x.isdigit() else 0):
            cd = data[budget][cid]
            if cd.get("status") != "success":
                continue
            iters = cd.get("iterations", "?")
            conv = "✓" if cd.get("converged") else "✗"
            dists = cd.get("spectral_distances", [])
            d_final = f"{dists[-1]:.4e}" if dists else "—"

            per_it = cd.get("per_iteration", [])
            states = per_it[-1].get("states", "?") if per_it else "?"
            actions = cd.get("total_actions", "?")

            bl = BUDGET_STYLE.get(budget, {}).get("label", budget)
            print(
                f"  {bl:<8} {cid:<6} {iters:>5} {conv:>5} "
                f"{d_final:>10} {states:>7} {actions:>8}"
            )

    print("═" * 75)

    # Average iterations per budget
    print("\n  Average iterations by budget:")
    for budget in ["1", "2", "3", "inf"]:
        if budget not in data:
            continue
        iter_counts = [
            cd["iterations"]
            for cd in data[budget].values()
            if cd.get("status") == "success" and "iterations" in cd
        ]
        if iter_counts:
            bl = BUDGET_STYLE.get(budget, {}).get("label", budget)
            print(f"    {bl}: {np.mean(iter_counts):.1f} ± {np.std(iter_counts):.1f}")


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot spectral convergence curves from Experiment 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_file",
        help="Path to experiment1_results.json",
    )
    parser.add_argument(
        "--fig-a-case",
        default=None,
        help="Case ID for Figure A (budget sweep). Default: largest case.",
    )
    parser.add_argument(
        "--fig-b-budget",
        default="2",
        help="Budget setting for Figure B (all cases). Default: 2.",
    )
    parser.add_argument(
        "--epsilon",
        type=float, default=0.05,
        help="Convergence threshold ε (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for figures (default: same as results file).",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip printing the summary table.",
    )

    args = parser.parse_args()
    results_path = Path(args.results_file)

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    data = load_results(results_path)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded results from {results_path}")
    print(f"Budgets: {list(data.keys())}")
    print(f"Output:  {output_dir}")

    # Determine default case for Figure A (pick the one with most iterations)
    fig_a_case = args.fig_a_case
    if fig_a_case is None:
        max_iters = 0
        for budget_data in data.values():
            for cid, cd in budget_data.items():
                n = cd.get("iterations", 0)
                if n > max_iters:
                    max_iters = n
                    fig_a_case = cid
        if fig_a_case is None:
            fig_a_case = "1"

    print(f"\nFigure A case: {fig_a_case}")
    print(f"Figure B budget: {args.fig_b_budget}")

    # ── Generate figures ──
    plot_figure_a(data, fig_a_case, args.epsilon, output_dir)
    plot_figure_b(data, args.fig_b_budget, args.epsilon, output_dir)
    plot_figure_c(data, fig_a_case, output_dir)

    # ── Summary table ──
    if not args.no_table:
        print_summary_table(data, args.epsilon)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
