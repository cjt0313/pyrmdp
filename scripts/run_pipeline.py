#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end domain robustification pipeline (Steps 0–6)

Given an RGB observation, task description, and number of robot policies,
produces a multi-policy PPDDL domain file.

Usage
-----
  # Minimal (requires OPENAI_API_KEY env-var with a vision-capable model):
  python scripts/run_pipeline.py \\
      --image scene.png \\
      --task "Pick up the red block and place it on the blue plate." \\
      --num-policies 3 \\
      --output-dir ./pipeline_output

  # With intermediate results also saved:
  python scripts/run_pipeline.py \\
      --image scene.png \\
      --task "Stack all blocks." \\
      --task "If a block falls, pick it up." \\
      --num-policies 3 \\
      --output-dir ./pipeline_output \\
      --save-intermediates

  # Use an existing PDDL domain instead of VLM (skip Step 0):
  python scripts/run_pipeline.py \\
      --domain domain.pddl \\
      --num-policies 3 \\
      --output-dir ./pipeline_output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure the package is importable when run from the repo root ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx  # noqa: F401  (kept for _save_graph helper)

from pyrmdp.synthesis.llm_config import LLMConfig, load_config, build_llm_fn
from pyrmdp.synthesis.config import PipelineConfig
from pyrmdp.synthesis.domain_genesis import generate_initial_domain
from pyrmdp.synthesis.iterative_synthesizer import IterativeDomainSynthesizer

# pyPPDDL parser
try:
    from pyppddl.ppddl.parser import load_domain, Domain
except ImportError:
    load_domain = None
    Domain = None


logger = logging.getLogger("pyrmdp.pipeline")


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    logger.info(f"  ✓ Saved: {path}")


def _save_json(path: Path, data: Any) -> None:
    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, frozenset):
            return sorted(obj)
        return str(obj)

    path.write_text(
        json.dumps(data, indent=2, default=_default), encoding="utf-8"
    )
    logger.info(f"  ✓ Saved: {path}")


def _save_graph(path: Path, graph: nx.DiGraph, label: str) -> None:
    """Save a NetworkX graph as a GraphML file."""
    # GraphML can't handle sets — convert edge/node attributes
    G = graph.copy()
    for _, _, d in G.edges(data=True):
        for k, v in list(d.items()):
            if isinstance(v, (set, frozenset)):
                d[k] = ",".join(sorted(str(x) for x in v))
            elif not isinstance(v, (str, int, float, bool)):
                d[k] = str(v)
    for _, d in G.nodes(data=True):
        for k, v in list(d.items()):
            if not isinstance(v, (str, int, float, bool)):
                d[k] = str(v)
    nx.write_graphml(G, str(path))
    logger.info(f"  ✓ Saved {label}: {path}")


class StepTimer:
    """Context manager that times a pipeline step."""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.elapsed = 0.0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *_):
        self.elapsed = time.time() - self.start
        logger.info(f"  ⏱  {self.step_name} completed in {self.elapsed:.1f}s")


# ════════════════════════════════════════════════════════════════════
#  Pipeline
# ════════════════════════════════════════════════════════════════════

def run_pipeline(
    *,
    image_paths: Optional[List[str]] = None,
    task_descriptions: Optional[List[str]] = None,
    domain_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    scene_description: str = "",
    domain_name_hint: Optional[str] = None,
) -> str:
    """
    Execute the full pipeline (Steps 0–6).

    Either provide (image_paths + task_descriptions) for Step 0,
    or provide domain_path to skip directly to the iterative loop.

    Steps 1–5 run in a spectral-convergence loop managed by
    ``IterativeDomainSynthesizer``.  Step 6 (multi-policy PPDDL
    emission) is included in the synthesizer's ``run()`` call.

    Parameters
    ----------
    config : PipelineConfig, optional
        Central configuration object.  If *None*, defaults are used.
        See ``pyrmdp.synthesis.config.PipelineConfig`` for all fields.

    Returns the final PPDDL domain string.
    """
    cfg = config or PipelineConfig()
    out = _ensure_dir(Path(cfg.output_dir))
    timings: Dict[str, float] = {}

    # ────────────────────────────────────────────────────────────────
    # Step 0: Domain Genesis (VLM + LLM)  — or load from file
    # ────────────────────────────────────────────────────────────────
    if domain_path:
        logger.info("═" * 60)
        logger.info("Skipping Step 0 — loading domain from file")
        logger.info("═" * 60)

        if load_domain is None:
            raise ImportError(
                "pyPPDDL is required to parse domain files.  "
                "pip install -e /path/to/pyPPDDL"
            )
        domain = load_domain(domain_path)
        pddl_str = Path(domain_path).read_text(encoding="utf-8")

        if cfg.save_intermediates:
            _save_text(out / "step0_domain_input.pddl", pddl_str)
    else:
        if not image_paths:
            raise ValueError("Provide --image or --domain.")
        if not task_descriptions:
            raise ValueError("Provide --task when using --image.")

        logger.info("═" * 60)
        logger.info("Step 0: Domain Genesis (VLM → types/predicates, LLM → operators)")
        logger.info("═" * 60)

        with StepTimer("Step 0") as t:
            result = generate_initial_domain(
                image_paths=[Path(p) for p in image_paths],
                task_descriptions=task_descriptions,
                scene_description=scene_description,
                domain_name_hint=domain_name_hint,
                return_parsed=True,
            )
        timings["step0"] = t.elapsed

        if isinstance(result, str):
            pddl_str = result
            if cfg.save_intermediates:
                _save_text(out / "step0_domain_generated.pddl", pddl_str)
            # Try to parse for downstream steps
            if load_domain is None:
                raise ImportError(
                    "pyPPDDL is required for Steps 1–6.  "
                    "pip install -e /path/to/pyPPDDL"
                )
            from pyppddl.ppddl.parser import load_domain_from_string
            domain = load_domain_from_string(pddl_str)
        else:
            domain = result
            pddl_str = "(generated domain — see parsed object)"

        if cfg.save_intermediates:
            _save_text(out / "step0_domain_generated.pddl", pddl_str)

        logger.info(
            f"  Domain: {domain.name}, "
            f"{len(domain.types)} types, "
            f"{len(domain.predicates)} predicates, "
            f"{len(domain.actions)} actions"
        )

    # ────────────────────────────────────────────────────────────────
    # Steps 1–5 (iterative loop) + Step 6 (emission)
    # ────────────────────────────────────────────────────────────────
    synth = IterativeDomainSynthesizer(
        domain,
        epsilon=cfg.epsilon,
        max_iterations=cfg.max_loop_iterations,
        failure_prob=cfg.failure_prob,
        scoring_config=cfg.scoring_config(),
        emission_config=cfg.emission_config(),
        output_dir=str(out),
        save_intermediates=cfg.save_intermediates,
    )

    ppddl_output = synth.run()

    # Print convergence summary
    summary = synth.summary()
    logger.info("")
    logger.info("Convergence summary:")
    logger.info(f"  Converged: {summary['converged']}")
    logger.info(f"  Iterations: {summary['iterations']}")
    if summary['spectral_distances']:
        logger.info(f"  Final Δ_spectral: {summary['spectral_distances'][-1]:.6f}")

    return ppddl_output


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="pyrmdp: End-to-end domain robustification pipeline (Steps 0–6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input sources (mutually optional: image+task OR domain) ──
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--image", "-i",
        action="append", dest="images", default=[],
        help="Path to RGB image(s) of the scene (repeatable). Triggers Step 0.",
    )
    input_group.add_argument(
        "--task", "-t",
        action="append", dest="tasks", default=[],
        help="Natural-language task description (repeatable). Required with --image.",
    )
    input_group.add_argument(
        "--domain", "-d",
        default=None,
        help="Path to an existing PDDL domain file (skips Step 0).",
    )
    input_group.add_argument(
        "--scene-description",
        default="",
        help="Optional extra scene description for the VLM (Step 0a).",
    )
    input_group.add_argument(
        "--domain-name",
        default=None,
        help="Suggested PDDL domain name (Step 0a).",
    )

    # ── Config file ──
    parser.add_argument(
        "--config", "-c",
        default=None,
        help=(
            "Path to a YAML config file (see PipelineConfig). "
            "CLI flags override values from the file. "
            "Generate a template with: python -c "
            "'from pyrmdp.synthesis.config import PipelineConfig; "
            "PipelineConfig.write_default()'"
        ),
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Write a default pipeline.yaml to the output dir and exit.",
    )

    # ── Output ──
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for all outputs (default: ./pipeline_output).",
    )
    output_group.add_argument(
        "--save-intermediates",
        action="store_true", default=None,
        help="Save intermediate results (per-step JSON/GraphML).",
    )

    # ── Pipeline parameter overrides ──
    params_group = parser.add_argument_group(
        "Pipeline parameters (override config file)"
    )
    params_group.add_argument(
        "--num-policies", "-K",
        type=int, default=None,
        help="Number of robot policy variants per action.",
    )
    params_group.add_argument(
        "--failure-prob",
        type=float, default=None,
        help="Probability of failure branch in Step 1.",
    )
    params_group.add_argument(
        "--max-delta-iterations",
        type=int, default=None,
        help="Max iterations for delta minimization in Step 5.",
    )
    params_group.add_argument(
        "--scoring-alpha",
        type=float, default=None,
        help="Weight for delta similarity in scoring (Step 5).",
    )
    params_group.add_argument(
        "--scoring-beta",
        type=float, default=None,
        help="Weight for topological gain in scoring (Step 5).",
    )
    params_group.add_argument(
        "--epsilon",
        type=float, default=None,
        help="Spectral-distance convergence threshold.",
    )
    params_group.add_argument(
        "--max-loop-iterations",
        type=int, default=None,
        help="Maximum iterations for the Steps 1-5 convergence loop.",
    )

    # ── Reward parameter overrides ──
    reward_group = parser.add_argument_group(
        "Reward parameters (override config file, Step 6)"
    )
    reward_group.add_argument("--success-reward", type=float, default=None)
    reward_group.add_argument("--unchanged-reward", type=float, default=None)
    reward_group.add_argument("--failure-reward", type=float, default=None)
    reward_group.add_argument("--human-reward", type=float, default=None)

    # ── Logging ──
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()

    # ── Configure logging ──
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Build PipelineConfig: file → CLI overrides ──
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        cfg = PipelineConfig()

    # Map CLI args → PipelineConfig field names
    cli_overrides = {
        "epsilon": args.epsilon,
        "max_loop_iterations": args.max_loop_iterations,
        "failure_prob": args.failure_prob,
        "scoring_alpha": args.scoring_alpha,
        "scoring_beta": args.scoring_beta,
        "max_delta_iterations": args.max_delta_iterations,
        "num_robot_policies": args.num_policies,
        "success_reward": args.success_reward,
        "unchanged_reward": args.unchanged_reward,
        "failure_reward": args.failure_reward,
        "human_reward": args.human_reward,
        "output_dir": args.output_dir,
        "save_intermediates": args.save_intermediates,
    }
    cfg = cfg.override_from_args(cli_overrides)

    # ── Dump-config mode ──
    if args.dump_config:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = cfg.write_yaml(out / "pipeline.yaml")
        print(f"Default config written to {p}")
        return

    # ── Validate inputs ──
    if not args.images and not args.domain:
        parser.error("Provide either --image (with --task) or --domain.")
    if args.images and not args.tasks:
        parser.error("--task is required when using --image.")

    # ── Run ──
    run_pipeline(
        image_paths=args.images or None,
        task_descriptions=args.tasks or None,
        domain_path=args.domain,
        config=cfg,
        scene_description=args.scene_description,
        domain_name_hint=args.domain_name,
    )


if __name__ == "__main__":
    main()
