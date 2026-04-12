"""
Step 6: Multi-Policy Operator Expansion → PPDDL Output

Transforms the final domain (base operators + synthesized recovery operators)
into a well-formed PPDDL file. For every operator, expands into K+1 versions:
  - K Robot Policies: probabilistic actions with reward annotations
  - 1 Human Policy: deterministic, 100% success, reward 0

Actions that went through Step 1 failure hallucination get 3 branches:
  - Success (nominal effect): reward +10
  - Unchanged (no effect):    reward -1
  - Worse (failure effect):   reward -10

Deterministic recovery operators (from Step 5, no failure branch) get 2 branches:
  - Success (nominal effect): reward +10
  - Unchanged (no effect):    reward -1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# pyPPDDL data model
try:
    from pyppddl.ppddl.parser import (
        ActionSchema,
        Domain,
        Effect,
        Predicate,
        TypedParam,
    )
except ImportError:
    raise ImportError(
        "pyPPDDL is required. Install it with: pip install -e /path/to/pyPPDDL"
    )

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════

@dataclass
class PolicyExpansionConfig:
    """Configuration for multi-policy operator expansion."""
    num_robot_policies: int = 3         # K robot policy variants
    success_reward: float = 10.0
    unchanged_reward: float = -1.0
    failure_reward: float = -10.0
    human_reward: float = 0.0
    # Priors for actions WITH a failure branch (3-branch: success/unchanged/worse)
    initial_success_prob: float = 1 / 3
    initial_unchanged_prob: float = 1 / 3
    initial_failure_prob: float = 1 / 3
    # Priors for actions WITHOUT a failure branch (2-branch: success/unchanged)
    deterministic_success_prob: float = 0.5
    deterministic_unchanged_prob: float = 0.5


# ════════════════════════════════════════════════════════════════════
#  PPDDL Serialization
# ════════════════════════════════════════════════════════════════════

def _indent(text: str, level: int = 1) -> str:
    """Indent each line of text by the given level (4 spaces per level)."""
    prefix = "    " * level
    return "\n".join(prefix + line for line in text.split("\n"))


def _format_typed_params(params: List[TypedParam]) -> str:
    """Format typed parameters as PDDL string."""
    if not params:
        return ""
    parts = []
    # Group by type for compact output
    by_type: Dict[str, List[str]] = {}
    for p in params:
        by_type.setdefault(p.type, []).append(p.name)
    for typ, names in by_type.items():
        names_str = " ".join(names)
        if typ == "object":
            parts.append(names_str)
        else:
            parts.append(f"{names_str} - {typ}")
    return " ".join(parts)


def _format_predicate_tuple(pred: Tuple) -> str:
    """Format a predicate tuple as PDDL: (pred_name arg1 arg2)."""
    return f"({' '.join(str(x) for x in pred)})"


def _format_precondition(precond: Any) -> str:
    """Format a precondition S-expression as PDDL string."""
    if precond is None:
        return "()"
    if isinstance(precond, list):
        inner = " ".join(_format_precondition(x) for x in precond)
        return f"({inner})"
    return str(precond)


def _format_effect(eff: Effect) -> str:
    """Format a single Effect as PDDL effect clauses."""
    parts = []
    for pred in eff.add_predicates:
        parts.append(_format_predicate_tuple(pred))
    for pred in eff.del_predicates:
        inner = _format_predicate_tuple(pred)
        parts.append(f"(not {inner})")
    for op, func_name, value in eff.numeric_effects:
        parts.append(f"({op} ({func_name}) {value})")

    if len(parts) == 0:
        return "()"
    if len(parts) == 1:
        return parts[0]
    return "(and\n" + "\n".join(f"    {p}" for p in parts) + "\n)"


# ════════════════════════════════════════════════════════════════════
#  Action Expansion
# ════════════════════════════════════════════════════════════════════

def _get_nominal_effect(action: ActionSchema) -> Effect:
    """Get the nominal (highest probability) effect from an action."""
    if not action.effects:
        return Effect(prob=1.0)
    # Find the effect with highest probability
    return max(action.effects, key=lambda e: e.prob)


def _get_failure_effect(action: ActionSchema) -> Optional[Effect]:
    """Get the failure (lowest probability) effect, if any."""
    if len(action.effects) <= 1:
        return None
    sorted_effs = sorted(action.effects, key=lambda e: e.prob)
    return sorted_effs[0]


def _make_unchanged_effect(reward_delta: float) -> Effect:
    """Create a no-op effect with only a reward change."""
    return Effect(
        prob=1.0,
        numeric_effects=[("increase", "reward", reward_delta)],
    )


def _expand_action_robot_policy(
    action: ActionSchema,
    policy_idx: int,
    config: PolicyExpansionConfig,
) -> ActionSchema:
    """
    Expand an action into a single robot-policy variant.

    Actions **with** a failure branch (from Step 1 hallucination) get
    3 probabilistic branches: success / unchanged / worse.

    Actions **without** a failure branch (deterministic recovery
    operators from Step 5) get 2 branches: success / unchanged.
    No dummy "worse" branch is emitted — the failure mode does not
    exist yet.
    """
    nominal = _get_nominal_effect(action)
    failure = _get_failure_effect(action)

    # Success branch: nominal effect + success reward
    if failure is not None:
        success_prob = config.initial_success_prob
    else:
        success_prob = config.deterministic_success_prob

    success_eff = Effect(prob=success_prob)
    success_eff.add_predicates = list(nominal.add_predicates)
    success_eff.del_predicates = list(nominal.del_predicates)
    success_eff.numeric_effects = list(nominal.numeric_effects) + [
        ("increase", "reward", config.success_reward)
    ]

    # Unchanged branch: no predicate changes, penalty reward
    if failure is not None:
        unchanged_prob = config.initial_unchanged_prob
    else:
        unchanged_prob = config.deterministic_unchanged_prob

    unchanged_eff = Effect(prob=unchanged_prob)
    unchanged_eff.numeric_effects = [
        ("increase", "reward", config.unchanged_reward)
    ]

    effects = [success_eff, unchanged_eff]

    # Worse branch: only if a failure effect exists
    if failure is not None:
        worse_eff = Effect(prob=config.initial_failure_prob)
        worse_eff.add_predicates = list(failure.add_predicates)
        worse_eff.del_predicates = list(failure.del_predicates)
        worse_eff.numeric_effects = list(failure.numeric_effects) + [
            ("increase", "reward", config.failure_reward)
        ]
        effects.append(worse_eff)

    return ActionSchema(
        name=f"{action.name}_robot{policy_idx}",
        parameters=list(action.parameters),
        precondition=action.precondition,
        effects=effects,
    )


def _expand_action_human_policy(
    action: ActionSchema,
    config: PolicyExpansionConfig,
) -> ActionSchema:
    """
    Expand an action into a human-policy variant:
    deterministic, 100% success, reward = 0.
    """
    nominal = _get_nominal_effect(action)

    human_eff = Effect(prob=1.0)
    human_eff.add_predicates = list(nominal.add_predicates)
    human_eff.del_predicates = list(nominal.del_predicates)
    # Keep only non-reward numeric effects, then add human_reward
    human_eff.numeric_effects = [
        ne for ne in nominal.numeric_effects
        if ne[1] not in ("reward", "total-reward")
    ] + [("increase", "reward", config.human_reward)]

    return ActionSchema(
        name=f"{action.name}_human",
        parameters=list(action.parameters),
        precondition=action.precondition,
        effects=[human_eff],
    )


# ════════════════════════════════════════════════════════════════════
#  Full PPDDL Emission
# ════════════════════════════════════════════════════════════════════

def _emit_domain_string(domain: Domain, expanded_actions: List[ActionSchema]) -> str:
    """Serialize a Domain + expanded actions into a PPDDL domain string."""
    lines = []
    lines.append(f"(define (domain {domain.name})")

    # Requirements
    reqs = list(domain.requirements)
    if ":probabilistic-effects" not in reqs:
        reqs.append(":probabilistic-effects")
    if ":numeric-fluents" not in reqs:
        reqs.append(":numeric-fluents")
    lines.append(f"  (:requirements {' '.join(reqs)})")

    # Types
    if domain.types:
        type_parts = []
        # Group by parent
        by_parent: Dict[str, List[str]] = {}
        for child, parent in domain.types.items():
            by_parent.setdefault(parent, []).append(child)
        for parent, children in by_parent.items():
            type_parts.append(f"    {' '.join(children)} - {parent}")
        lines.append("  (:types")
        lines.extend(type_parts)
        lines.append("  )")

    # Predicates
    if domain.predicates:
        lines.append("  (:predicates")
        for pred in domain.predicates:
            params_str = _format_typed_params(pred.parameters)
            if params_str:
                lines.append(f"    ({pred.name} {params_str})")
            else:
                lines.append(f"    ({pred.name})")
        lines.append("  )")

    # Functions
    if domain.functions:
        lines.append("  (:functions")
        for func in domain.functions:
            params_str = _format_typed_params(func.parameters)
            if params_str:
                lines.append(f"    ({func.name} {params_str})")
            else:
                lines.append(f"    ({func.name})")
        lines.append("  )")
    elif any(
        ne
        for a in expanded_actions
        for e in a.effects
        for ne in e.numeric_effects
    ):
        # Need reward function
        lines.append("  (:functions")
        lines.append("    (reward)")
        lines.append("  )")

    # Actions
    for action in expanded_actions:
        lines.append("")
        lines.append(f"  (:action {action.name}")
        params_str = _format_typed_params(action.parameters)
        lines.append(f"    :parameters ({params_str})")

        precond_str = _format_precondition(action.precondition)
        lines.append(f"    :precondition {precond_str}")

        # Effects
        if len(action.effects) == 1:
            eff_str = _format_effect(action.effects[0])
            lines.append(f"    :effect {eff_str}")
        else:
            # Probabilistic
            lines.append("    :effect (probabilistic")
            for eff in action.effects:
                eff_str = _format_effect(eff)
                lines.append(f"        {eff.prob} {eff_str}")
            lines.append("    )")

        lines.append("  )")

    lines.append(")")
    return "\n".join(lines)


def emit_ppddl(
    domain: Domain,
    *,
    output_path: Optional[str] = None,
    config: Optional[PolicyExpansionConfig] = None,
) -> str:
    """
    Expand all actions in the domain into multi-policy PPDDL and emit.

    For each original action, generates:
      - K robot policy variants (probabilistic, 3 branches each)
      - 1 human policy variant (deterministic, 100% success)

    Parameters
    ----------
    domain : Domain
        The augmented pyPPDDL Domain (with failure-hallucinated + synthesized actions).
    output_path : str, optional
        If given, writes the PPDDL string to this file.
    config : PolicyExpansionConfig, optional
        Expansion parameters. Uses defaults if None.

    Returns
    -------
    str
        The complete PPDDL domain file content.
    """
    if config is None:
        config = PolicyExpansionConfig()

    expanded_actions: List[ActionSchema] = []

    for action in domain.actions:
        # K robot policy variants
        for k in range(1, config.num_robot_policies + 1):
            robot_action = _expand_action_robot_policy(action, k, config)
            expanded_actions.append(robot_action)

        # 1 human policy variant
        human_action = _expand_action_human_policy(action, config)
        expanded_actions.append(human_action)

    logger.info(
        f"Expanded {len(domain.actions)} actions → "
        f"{len(expanded_actions)} multi-policy actions "
        f"({config.num_robot_policies} robot + 1 human per action)"
    )

    ppddl_str = _emit_domain_string(domain, expanded_actions)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ppddl_str)
        logger.info(f"PPDDL domain written to: {output_path}")

    return ppddl_str
