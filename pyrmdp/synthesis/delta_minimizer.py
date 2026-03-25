"""
Step 5: Delta Minimization & Operator Synthesis

Iteratively selects the best (sink → source) SCC pair to bridge,
minimizing logical Hamming distance (predicate delta) while maximizing
topological gain. Queries an LLM to synthesize the bridging PPDDL operator
plus its failure mode. Repeats until the DAG collapses to a single SCC.

Scoring: α·(1 − norm_delta) + β·(norm_gain), default α=0.7, β=0.3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from .graph_analysis import (
    AugmentationBound,
    CondensationResult,
    compute_augmentation_bound,
    condense_to_dag,
    get_scc_representative_predicates,
)

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
#  Config & Data Classes
# ════════════════════════════════════════════════════════════════════

@dataclass
class ScoringConfig:
    """Configuration for the delta minimization scoring function."""
    alpha: float = 0.7              # Weight for delta (lower delta → higher score)
    beta: float = 0.3               # Weight for topological gain
    max_iterations: int = 50        # Max synthesis iterations
    max_candidates_per_iter: int = 10  # Candidates to try per iteration
    delta_threshold: int = 15       # Max predicates LLM can handle


@dataclass
class CandidateEdge:
    """A candidate edge from a sink SCC to a source SCC."""
    sink_scc: int
    source_scc: int
    sink_true_preds: Set[str]
    sink_false_preds: Set[str]
    source_true_preds: Set[str]
    source_false_preds: Set[str]
    delta: int
    topological_gain: int
    weighted_score: float = 0.0


@dataclass
class SynthesizedOperator:
    """An operator synthesized by the LLM to bridge two SCCs."""
    name: str
    parameters: List[TypedParam]
    precondition_preds: List[str]
    nominal_add: List[Tuple]
    nominal_del: List[Tuple]
    failure_add: List[Tuple]
    failure_del: List[Tuple]
    numeric_effects: List[Tuple]
    source_scc: int
    sink_scc: int
    delta: int
    action_schema: Optional[ActionSchema] = None


@dataclass
class DeltaMinimizationResult:
    """Result of the delta minimization loop."""
    operators: List[SynthesizedOperator]
    stats: Dict[str, Any]
    final_sources: List[int]
    final_sinks: List[int]
    is_irreducible: bool


# ════════════════════════════════════════════════════════════════════
#  Logical Hamming Distance
# ════════════════════════════════════════════════════════════════════

def calculate_logical_hamming_distance(
    true_preds_u: Set[str],
    false_preds_u: Set[str],
    true_preds_v: Set[str],
    false_preds_v: Set[str],
) -> int:
    """
    Calculate the logical Hamming distance between two abstract states.

    Counts predicates that must change truth value to get from state U
    to state V:
      - predicates true in U but false in V (must become false)
      - predicates false in U but true in V (must become true)
      - predicates true in V but unknown in U (must become true)
      - predicates true in U but unknown in V (may need to change)
    """
    # Predicates that need to flip from true → false
    must_delete = true_preds_u & false_preds_v
    # Predicates that need to flip from false → true
    must_add = false_preds_u & true_preds_v
    # Predicates true in target but not mentioned in source
    new_true = true_preds_v - true_preds_u - false_preds_u
    # Predicates false in target but not mentioned in source
    new_false = false_preds_v - true_preds_u - false_preds_u

    return len(must_delete) + len(must_add) + len(new_true) + len(new_false)


# ════════════════════════════════════════════════════════════════════
#  LLM Operator Synthesis
# ════════════════════════════════════════════════════════════════════

def _build_synthesis_prompt(candidate: CandidateEdge) -> str:
    """Build the LLM prompt for operator synthesis."""
    return f"""You are a PDDL domain expert. Synthesize a PPDDL operator that
transitions the world from State U (a sink state) to State V (a source state).

State U (Sink SCC {candidate.sink_scc}):
  True predicates:  {sorted(candidate.sink_true_preds)}
  False predicates: {sorted(candidate.sink_false_preds)}

State V (Source SCC {candidate.source_scc}):
  True predicates:  {sorted(candidate.source_true_preds)}
  False predicates: {sorted(candidate.source_false_preds)}

Required change (delta): {candidate.delta} predicates must change.

Requirements:
1. Create a valid PPDDL operator with typed parameters, preconditions, and effects.
2. The precondition should be satisfiable in State U.
3. The nominal effect should produce State V (or move towards it).
4. Hallucinate ONE physically plausible "worse effect" (failure mode).
5. Keep the operator generalizable with parametrized variables.

Respond with ONLY valid JSON:
{{
    "name": "recover_action_name",
    "parameters": [{{"name": "?obj", "type": "physical-item"}}],
    "preconditions": [["pred_name", "?arg1", "?arg2"]],
    "nominal_add": [["pred_name", "?arg1"]],
    "nominal_del": [["pred_name", "?arg1"]],
    "failure_add": [["pred_name", "?arg1"]],
    "failure_del": [["pred_name", "?arg1"]],
    "numeric_effects": [["decrease", "reward", 1]],
    "explanation": "This recovery action does X."
}}
"""


def _parse_synthesis_response(
    response_text: str,
    candidate: CandidateEdge,
) -> Optional[SynthesizedOperator]:
    """Parse the LLM JSON response into a SynthesizedOperator."""
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        data = json.loads(text[start:end])
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Failed to parse LLM synthesis response: {e}")
        return None

    # Build parameters
    params = [
        TypedParam(name=p["name"], type=p.get("type", "object"))
        for p in data.get("parameters", [])
    ]

    # Build precondition predicate names
    precond_preds = [
        p[0] if isinstance(p, list) else p
        for p in data.get("preconditions", [])
    ]

    return SynthesizedOperator(
        name=data.get("name", f"recover_{candidate.sink_scc}_to_{candidate.source_scc}"),
        parameters=params,
        precondition_preds=precond_preds,
        nominal_add=[tuple(p) for p in data.get("nominal_add", [])],
        nominal_del=[tuple(p) for p in data.get("nominal_del", [])],
        failure_add=[tuple(p) for p in data.get("failure_add", [])],
        failure_del=[tuple(p) for p in data.get("failure_del", [])],
        numeric_effects=[tuple(n) for n in data.get("numeric_effects", [])],
        source_scc=candidate.source_scc,
        sink_scc=candidate.sink_scc,
        delta=candidate.delta,
    )


def _synthesize_operator(
    candidate: CandidateEdge,
    llm_fn: Callable[[str], str],
) -> Optional[SynthesizedOperator]:
    """Query the LLM to synthesize a PPDDL operator for a candidate edge."""
    prompt = _build_synthesis_prompt(candidate)
    try:
        response = llm_fn(prompt)
        return _parse_synthesis_response(response, candidate)
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        return None


def _convert_to_action_schema(
    op: SynthesizedOperator,
    nominal_prob: float = 0.9,
    failure_prob: float = 0.1,
) -> ActionSchema:
    """Convert a SynthesizedOperator into a pyPPDDL ActionSchema."""
    # Build precondition S-expression
    precond_atoms = []
    for pred_name in op.precondition_preds:
        # Use the first parameter as a generic variable
        precond_atoms.append([pred_name] + [p.name for p in op.parameters[:1]])

    if len(precond_atoms) == 0:
        precondition = None
    elif len(precond_atoms) == 1:
        precondition = precond_atoms[0]
    else:
        precondition = ["and"] + precond_atoms

    # Build effects
    # Nominal effect
    nominal_eff = Effect(prob=nominal_prob)
    nominal_eff.add_predicates = list(op.nominal_add)
    nominal_eff.del_predicates = list(op.nominal_del)
    nominal_eff.numeric_effects = list(op.numeric_effects)

    # Failure effect
    failure_eff = Effect(prob=failure_prob)
    failure_eff.add_predicates = list(op.failure_add)
    failure_eff.del_predicates = list(op.failure_del)

    action = ActionSchema(
        name=op.name,
        parameters=list(op.parameters),
        precondition=precondition,
        effects=[nominal_eff, failure_eff],
    )
    op.action_schema = action
    return action


# ════════════════════════════════════════════════════════════════════
#  Main Delta Minimization Loop
# ════════════════════════════════════════════════════════════════════

def delta_minimize(
    abstract_graph: nx.DiGraph,
    domain: Domain,
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
    config: Optional[ScoringConfig] = None,
    nominal_prob: float = 0.9,
    failure_prob: float = 0.1,
) -> DeltaMinimizationResult:
    """
    Iteratively synthesize recovery operators to make the abstract
    transition graph strongly connected (irreducible).

    Parameters
    ----------
    abstract_graph : nx.DiGraph
        The abstract transition graph from Step 2.
    domain : Domain
        The pyPPDDL Domain (will be mutated with new actions).
    llm_fn : callable, optional
        Function(prompt: str) -> str.  If None, builds one from
        ``llm.yaml`` / env-vars via :func:`llm_config.build_llm_fn`.
    config : ScoringConfig, optional
        Scoring parameters. Uses defaults if None.
    nominal_prob : float
        Probability for the nominal (success) effect of synthesized operators.
    failure_prob : float
        Probability for the failure effect of synthesized operators.

    Returns
    -------
    DeltaMinimizationResult
    """
    if config is None:
        config = ScoringConfig()

    if llm_fn is None:
        from .llm_config import build_llm_fn
        llm_fn = build_llm_fn()

    operators: List[SynthesizedOperator] = []
    stats = {"successful": 0, "failed": 0, "deltas": [], "iterations": 0}

    # Work on a copy of the graph to add edges
    working_graph = abstract_graph.copy()

    for iteration in range(config.max_iterations):
        stats["iterations"] = iteration + 1

        # Re-condense and check
        condensation = condense_to_dag(working_graph)
        aug_bound = compute_augmentation_bound(condensation)

        if aug_bound.is_already_irreducible:
            logger.info(f"Graph is irreducible after {iteration} iterations!")
            break

        logger.info(
            f"Iteration {iteration + 1}: "
            f"sources={len(aug_bound.sources)}, sinks={len(aug_bound.sinks)}"
        )

        # ── Generate candidates: sink → source pairs ──
        candidates: List[CandidateEdge] = []

        for sink_id in aug_bound.sinks:
            sink_true, sink_false = get_scc_representative_predicates(
                sink_id, condensation
            )
            for src_id in aug_bound.sources:
                if sink_id == src_id:
                    continue

                src_true, src_false = get_scc_representative_predicates(
                    src_id, condensation
                )

                delta = calculate_logical_hamming_distance(
                    sink_true, sink_false, src_true, src_false
                )

                if delta > config.delta_threshold:
                    continue

                # Topological gain heuristic
                gain = 0
                gain += 1 if sink_id in aug_bound.sinks else 0
                gain += 1 if src_id in aug_bound.sources else 0

                candidates.append(CandidateEdge(
                    sink_scc=sink_id,
                    source_scc=src_id,
                    sink_true_preds=sink_true,
                    sink_false_preds=sink_false,
                    source_true_preds=src_true,
                    source_false_preds=src_false,
                    delta=delta,
                    topological_gain=gain,
                ))

        if not candidates:
            logger.warning("No valid candidates remaining. Stopping.")
            break

        # ── Score candidates ──
        max_delta = max(c.delta for c in candidates) or 1
        max_gain = max(c.topological_gain for c in candidates) or 1

        for c in candidates:
            norm_delta = c.delta / max_delta
            norm_gain = c.topological_gain / max_gain
            c.weighted_score = (
                config.alpha * (1.0 - norm_delta)
                + config.beta * norm_gain
            )

        candidates.sort(key=lambda c: c.weighted_score, reverse=True)

        # ── Try top candidates ──
        operator_added = False
        for i, candidate in enumerate(candidates[: config.max_candidates_per_iter]):
            logger.info(
                f"  Trying #{i + 1}: SCC-{candidate.sink_scc} → "
                f"SCC-{candidate.source_scc} (Δ={candidate.delta}, "
                f"gain={candidate.topological_gain}, "
                f"score={candidate.weighted_score:.3f})"
            )

            op = _synthesize_operator(candidate, llm_fn)
            if op is None:
                stats["failed"] += 1
                continue

            # Convert to ActionSchema and add to domain
            action_schema = _convert_to_action_schema(
                op, nominal_prob=nominal_prob, failure_prob=failure_prob
            )
            domain.actions.append(action_schema)
            operators.append(op)
            stats["successful"] += 1
            stats["deltas"].append(candidate.delta)

            # Add edge to working graph (connect representative members)
            sink_members = condensation.scc_state_map.get(candidate.sink_scc, [])
            src_members = condensation.scc_state_map.get(candidate.source_scc, [])
            if sink_members and src_members:
                working_graph.add_edge(
                    sink_members[0],
                    src_members[0],
                    action=op.name,
                    prob=nominal_prob,
                )

            operator_added = True
            logger.info(f"    ✓ Synthesized: {op.name}")
            break

        if not operator_added:
            logger.warning("LLM failed on all candidates this iteration. Stopping.")
            break

    # Final check
    final_condensation = condense_to_dag(working_graph)
    final_bound = compute_augmentation_bound(final_condensation)

    avg_delta = (
        sum(stats["deltas"]) / len(stats["deltas"]) if stats["deltas"] else 0
    )
    success_rate = (
        stats["successful"] / (stats["successful"] + stats["failed"])
        if (stats["successful"] + stats["failed"]) > 0
        else 0
    )

    logger.info(
        f"Delta minimization complete: {len(operators)} operators synthesized, "
        f"avg delta={avg_delta:.1f}, success rate={success_rate:.1%}"
    )

    return DeltaMinimizationResult(
        operators=operators,
        stats=stats,
        final_sources=final_bound.sources,
        final_sinks=final_bound.sinks,
        is_irreducible=final_bound.is_already_irreducible,
    )
