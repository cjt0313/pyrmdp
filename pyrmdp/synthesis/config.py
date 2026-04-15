"""
Centralised pipeline configuration for the synthesis pipeline.

All technical parameters for Steps 1–6 live here in a single
``PipelineConfig`` dataclass.  It can be:

* Constructed directly in Python
* Loaded from a YAML file (``PipelineConfig.from_yaml("pipeline.yaml")``)
* Saved to YAML for reproducibility
* Written as a default template (``PipelineConfig.write_default()``)

CLI flags in ``run_pipeline.py`` override values loaded from the file.

Usage
-----
>>> from pyrmdp.synthesis.config import PipelineConfig
>>> cfg = PipelineConfig()                       # all defaults
>>> cfg = PipelineConfig.from_yaml("my.yaml")    # from file
>>> cfg.write_yaml("pipeline.yaml")              # save for repro
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    All tuneable pipeline parameters, grouped by step.

    Provides factory methods ``.scoring_config()`` and
    ``.emission_config()`` that return the step-specific dataclasses
    expected by ``delta_minimize()`` and ``emit_ppddl()``.
    """

    # ── Iterative convergence loop ───────────────────────────────
    epsilon: float = 0.1
    """Wasserstein spectral-distance threshold Δ_W < ε to declare convergence.
    Wasserstein distance is dimension-invariant and provides the tightest signal."""

    max_loop_iterations: int = 10
    """Hard cap on outer loop iterations (Steps 1→5)."""

    # ── Step 1: Failure Hallucination ────────────────────────────
    failure_prob: float = 0.1
    """Probability assigned to the failure branch of each action."""

    # ── Step 2: Abstract State Pruning ────────────────────────────
    enable_mutex_pruning: bool = True
    """Enable R5 (LLM-based mutex pruning) after abstract state enumeration."""

    # ── Step 5: Delta Minimization ───────────────────────────────
    scoring_alpha: float = 0.7
    """Weight for delta similarity (lower delta → higher score)."""

    scoring_beta: float = 0.3
    """Weight for topological gain in the scoring function."""

    max_delta_iterations: int = 50
    """Max iterations inside the delta-minimizer per loop pass."""

    max_recovery_per_iter: Optional[int] = None
    """Budget cap on recovery operators per outer-loop iteration (None = unlimited)."""

    max_candidates_per_iter: int = 10
    """Candidates to try per delta-minimization iteration."""

    delta_threshold: int = 15
    """Max predicate delta the LLM can handle in one synthesis prompt."""

    # ── Step 6: Multi-Policy Emission ────────────────────────────
    num_robot_policies: int = 3
    """K robot-policy variants per action."""

    success_reward: float = 10.0
    """Reward for the success (nominal-effect) branch."""

    unchanged_reward: float = -1.0
    """Reward for the unchanged (no-effect) branch."""

    failure_reward: float = -10.0
    """Reward for the failure-effect branch."""

    human_reward: float = 0.0
    """Reward for the deterministic human-policy variant."""

    initial_success_prob: float = 1.0 / 3
    """Initial probability of the success branch (uniform prior)."""

    initial_unchanged_prob: float = 1.0 / 3
    """Initial probability of the unchanged branch."""

    initial_failure_prob: float = 1.0 / 3
    """Initial probability of the failure branch."""

    # ── Output ───────────────────────────────────────────────────
    output_dir: str = "./pipeline_output"
    """Directory for all outputs (PPDDL + intermediates)."""

    save_intermediates: bool = False
    """Whether to persist per-step JSON/GraphML alongside the PPDDL."""

    visualize: bool = True
    """Generate interactive evolution.html visualization after the pipeline."""

    # ══════════════════════════════════════════════════════════════
    #  Factory methods → step-specific config objects
    # ══════════════════════════════════════════════════════════════

    def scoring_config(self):
        """Build a ``ScoringConfig`` for Step 5 delta minimization."""
        from .delta_minimizer import ScoringConfig

        return ScoringConfig(
            alpha=self.scoring_alpha,
            beta=self.scoring_beta,
            max_iterations=self.max_delta_iterations,
            max_candidates_per_iter=self.max_candidates_per_iter,
            delta_threshold=self.delta_threshold,
            max_recovery_per_iter=self.max_recovery_per_iter,
        )

    def emission_config(self):
        """Build a ``PolicyExpansionConfig`` for Step 6 PPDDL emission."""
        from .ppddl_emitter import PolicyExpansionConfig

        return PolicyExpansionConfig(
            num_robot_policies=self.num_robot_policies,
            success_reward=self.success_reward,
            unchanged_reward=self.unchanged_reward,
            failure_reward=self.failure_reward,
            human_reward=self.human_reward,
            initial_success_prob=self.initial_success_prob,
            initial_unchanged_prob=self.initial_unchanged_prob,
            initial_failure_prob=self.initial_failure_prob,
        )

    # ══════════════════════════════════════════════════════════════
    #  Serialisation
    # ══════════════════════════════════════════════════════════════

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict of all fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Build from a dict, silently ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ── YAML helpers (soft dependency) ────────────────────────────

    def write_yaml(self, path: str | Path) -> Path:
        """
        Serialize this config to a YAML file.

        Raises ``ImportError`` if PyYAML is not installed.
        """
        import yaml  # noqa: F811

        p = Path(path)
        sections = {
            "# ── Iterative convergence loop": [
                "epsilon", "max_loop_iterations",
            ],
            "# ── Step 1: Failure Hallucination": [
                "failure_prob",
            ],
            "# ── Step 2: Abstract State Pruning": [
                "enable_mutex_pruning",
            ],
            "# ── Step 5: Delta Minimization": [
                "scoring_alpha", "scoring_beta", "max_delta_iterations",
                "max_recovery_per_iter", "max_candidates_per_iter",
                "delta_threshold",
            ],
            "# ── Step 6: Multi-Policy Emission": [
                "num_robot_policies",
                "success_reward", "unchanged_reward",
                "failure_reward", "human_reward",
                "initial_success_prob", "initial_unchanged_prob",
                "initial_failure_prob",
            ],
            "# ── Output": [
                "output_dir", "save_intermediates",
            ],
        }

        d = self.to_dict()
        lines = ["# pyrmdp pipeline configuration", ""]
        for header, keys in sections.items():
            lines.append(header)
            chunk = {k: d[k] for k in keys}
            lines.append(yaml.dump(chunk, default_flow_style=False).strip())
            lines.append("")

        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Config saved: {p}")
        return p

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """
        Load config from a YAML file.

        Raises ``FileNotFoundError`` if the file does not exist.
        Raises ``ImportError`` if PyYAML is not installed.
        """
        import yaml  # noqa: F811

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        logger.info(f"Config loaded: {p} ({len(data)} keys)")
        return cls.from_dict(data)

    @classmethod
    def write_default(cls, path: str | Path = "pipeline.yaml") -> Path:
        """Write a template config file with all default values."""
        return cls().write_yaml(path)

    # ══════════════════════════════════════════════════════════════
    #  CLI helpers
    # ══════════════════════════════════════════════════════════════

    def override_from_args(self, args: Dict[str, Any]) -> "PipelineConfig":
        """
        Return a *new* config with non-None values from *args* merged
        on top of this one.  Used to let CLI flags override a YAML base.
        """
        d = self.to_dict()
        for k, v in args.items():
            if v is not None and k in d:
                d[k] = v
        return PipelineConfig.from_dict(d)
