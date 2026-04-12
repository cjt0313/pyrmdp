"""
Step 5: Delta Minimization & Deterministic Recovery Synthesis

Iteratively selects the best (sink -> source) SCC pair to bridge,
minimizing logical Hamming distance (predicate delta) while maximizing
topological gain.  For each selected pair the recovery operator is
computed **deterministically** from the predicate delta -- no LLM call.

  * Preconditions: minimal causal subset of the sink state (only
    predicates sharing a unified variable with the delta effects).
  * Effects: single deterministic outcome (prob 1.0) that adds/deletes
    exactly the predicates in the delta.
  * No failure branch, no numeric rewards -- those are injected later
    by Step 1 (failure hallucination) and Step 6 (PPDDL emission).

Scoring: alpha*(1 - norm_delta) + beta*(norm_gain), default alpha=0.7, beta=0.3.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
    max_recovery_per_iter: Optional[int] = None  # Budget cap (None = unlimited)


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
    """A deterministic recovery operator bridging two SCCs.

    Built purely from the predicate delta between sink and source
    states — no LLM involved.
    """
    name: str
    parameters: List[TypedParam]
    precondition_preds: List[str]        # bare predicate names in precondition
    precondition_atoms: List[Tuple]      # grounded atoms, e.g. ("holding", "?r", "?x")
    nominal_add: List[Tuple]             # add-effects as grounded tuples
    nominal_del: List[Tuple]             # del-effects as grounded tuples
    failure_add: List[Tuple] = field(default_factory=list)   # always empty (Step 1 adds later)
    failure_del: List[Tuple] = field(default_factory=list)   # always empty
    numeric_effects: List[Tuple] = field(default_factory=list)  # always empty (Step 6 adds)
    source_scc: int = 0
    sink_scc: int = 0
    delta: int = 0
    sink_node: str = ""
    source_node: str = ""
    action_schema: Optional[ActionSchema] = None


@dataclass
class DeltaMinimizationResult:
    """Result of the delta minimization loop."""
    operators: List[SynthesizedOperator]
    stats: Dict[str, Any]
    final_sources: List[int]
    final_sinks: List[int]
    is_irreducible: bool
    augmented_graph: Optional[nx.DiGraph] = None


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


def mutex_aware_hamming_distance(
    true_preds_u: Set[str],
    false_preds_u: Set[str],
    true_preds_v: Set[str],
    false_preds_v: Set[str],
    mutex_groups: Optional[List[Any]] = None,
) -> int:
    """
    Mutex-corrected logical Hamming distance between two abstract states.

    Standard Hamming counts switching from ``opened`` to ``closed`` as
    δ=2 (one add + one delete).  But if ``{opened, closed}`` form an
    exactly-one group, the switch is really **one** atomic change → δ=1.

    Algorithm:
      1. Compute standard add/delete sets.
      2. For each mutex group, greedily match a *must_add* member with
         a *must_delete* member.  Each match reduces distance by 1.

    Parameters
    ----------
    true_preds_u, false_preds_u : set[str]
        Source state truth assignment.
    true_preds_v, false_preds_v : set[str]
        Target state truth assignment.
    mutex_groups : list[ExactlyOneGroup], optional
        Exactly-one mutex groups.  If *None* or empty, falls back to
        the standard distance.

    Returns
    -------
    int
        Corrected distance.
    """
    # Standard components
    must_delete = true_preds_u & false_preds_v
    must_add = false_preds_u & true_preds_v
    new_true = true_preds_v - true_preds_u - false_preds_u
    new_false = false_preds_v - true_preds_u - false_preds_u

    base = len(must_delete) + len(must_add) + len(new_true) + len(new_false)

    if not mutex_groups:
        return base

    # Greedy pairing: for each group, if one member is in must_add AND
    # another is in must_delete, they form a single swap → save 1.
    savings = 0
    paired_add: Set[str] = set()
    paired_del: Set[str] = set()

    for grp in mutex_groups:
        group_preds = set(grp.predicates)
        add_in_group = (must_add & group_preds) - paired_add
        del_in_group = (must_delete & group_preds) - paired_del

        # Pair them 1:1 greedily
        pairs = min(len(add_in_group), len(del_in_group))
        if pairs > 0:
            # Mark as paired (take arbitrary elements)
            for a, d in zip(sorted(add_in_group), sorted(del_in_group)):
                paired_add.add(a)
                paired_del.add(d)
                savings += 1
                if len(paired_add) >= pairs:
                    break

    return base - savings


# ════════════════════════════════════════════════════════════════════
#  Deterministic Recovery Operator Synthesis
# ════════════════════════════════════════════════════════════════════

def _build_pred_lookup(domain: Domain) -> Dict[str, "Predicate"]:
    """Build ``{predicate_name: Predicate}`` from the domain."""
    return {p.name: p for p in domain.predicates}


def _ground_predicate(
    pred_name: str,
    pred_lookup: Dict[str, "Predicate"],
    var_pool: "OrderedDict[str, str]",
) -> Tuple:
    """Ground a bare predicate name into a fully-typed tuple.

    For each parameter slot in the domain predicate signature, reuse an
    existing variable from *var_pool* if the same ``(name, type)`` was
    already registered, otherwise mint a fresh ``?v<N>`` variable.

    Returns
    -------
    tuple
        ``(pred_name, var1, var2, ...)`` ready for PPDDL add/del lists.
    """
    pred_def = pred_lookup.get(pred_name)
    if pred_def is None:
        # Unknown predicate — treat as 0-arity.
        logger.warning(f"Predicate '{pred_name}' not found in domain; treating as 0-arity")
        return (pred_name,)

    args: List[str] = []
    for param in pred_def.parameters:
        # Reuse an existing variable with the same PDDL name.
        # Domain predicate signatures already use canonical variable
        # names (e.g. ?r - robot, ?x - movable) which encode both
        # the intended role and the type.  Re-using them directly
        # ensures unification across predicates that share the same
        # typed slot.
        var_name = param.name
        if var_name not in var_pool:
            var_pool[var_name] = param.type
        args.append(var_name)
    return (pred_name, *args)


def _synthesize_operator(
    candidate: CandidateEdge,
    domain: Domain,
) -> SynthesizedOperator:
    """Deterministically synthesize a recovery operator from the delta.

    Algorithm
    ---------
    1. Compute the **delta** between sink and source states:
       - ``must_add``  = predicates that must become true
       - ``must_del``  = predicates that must become false
    2. **Ground** each predicate in the delta using the domain's predicate
       signatures, building a unified variable pool.
    3. Compute **minimal causal preconditions**: only sink-true predicates
       whose grounded form shares ≥ 1 variable with the delta effects.
    4. Return a :class:`SynthesizedOperator` with no failure branch and
       no numeric effects.
    """
    pred_lookup = _build_pred_lookup(domain)

    # ── 1. Compute the delta ──
    #  must_add: false (or unknown) in sink, true in source
    must_add = (
        (candidate.source_true_preds - candidate.sink_true_preds)
    )
    #  must_del: true in sink, false (or unknown) in source
    must_del = (
        (candidate.sink_true_preds - candidate.source_true_preds)
        & (candidate.source_false_preds | (
            candidate.sink_true_preds - candidate.source_true_preds
            - candidate.source_false_preds
        ))
    )
    # More precisely: delete anything true in sink that is NOT true in
    # source.  If it appears explicitly false in source, definitely
    # delete.  If it is simply absent from source ("don't care"), we
    # still want the transition to land in source, so we delete.
    must_del = candidate.sink_true_preds - candidate.source_true_preds

    # ── 2. Ground delta predicates (builds unified var_pool) ──
    var_pool: OrderedDict[str, str] = OrderedDict()  # var_name → type

    add_atoms: List[Tuple] = []
    for pname in sorted(must_add):
        add_atoms.append(_ground_predicate(pname, pred_lookup, var_pool))

    del_atoms: List[Tuple] = []
    for pname in sorted(must_del):
        del_atoms.append(_ground_predicate(pname, pred_lookup, var_pool))

    # ── 3. Collect variables that appear in the delta ──
    delta_vars: Set[str] = set()
    for atom in add_atoms + del_atoms:
        delta_vars.update(atom[1:])  # skip predicate name

    # ── 4. Minimal causal preconditions ──
    # Ground every sink-true predicate, but keep only those sharing
    # at least one variable with the delta effects.
    precond_atoms: List[Tuple] = []
    precond_preds: List[str] = []
    for pname in sorted(candidate.sink_true_preds):
        atom = _ground_predicate(pname, pred_lookup, var_pool)
        atom_vars = set(atom[1:])
        if atom_vars & delta_vars:
            precond_atoms.append(atom)
            precond_preds.append(pname)

    # ── 5. Build typed parameter list from var_pool ──
    parameters = [
        TypedParam(name=vname, type=vtype)
        for vname, vtype in var_pool.items()
        # Only include vars that actually appear in precond + effects
        if vname in delta_vars or any(
            vname in atom[1:] for atom in precond_atoms
        )
    ]

    name = f"recover_{candidate.sink_scc}_to_{candidate.source_scc}"
    return SynthesizedOperator(
        name=name,
        parameters=parameters,
        precondition_preds=precond_preds,
        precondition_atoms=precond_atoms,
        nominal_add=add_atoms,
        nominal_del=del_atoms,
        source_scc=candidate.source_scc,
        sink_scc=candidate.sink_scc,
        delta=candidate.delta,
    )


def _convert_to_action_schema(op: SynthesizedOperator) -> ActionSchema:
    """Convert a SynthesizedOperator into a deterministic ActionSchema.

    Single effect at probability 1.0, no failure branch, no numeric
    effects.  Step 1 (failure hallucination) and Step 6 (PPDDL
    emission) will layer on probabilistic branches and rewards later.
    """
    # Build precondition S-expression from grounded atoms
    if len(op.precondition_atoms) == 0:
        precondition = None
    elif len(op.precondition_atoms) == 1:
        precondition = list(op.precondition_atoms[0])
    else:
        precondition = ["and"] + [list(a) for a in op.precondition_atoms]

    # Single deterministic effect
    eff = Effect(prob=1.0)
    eff.add_predicates = list(op.nominal_add)
    eff.del_predicates = list(op.nominal_del)

    action = ActionSchema(
        name=op.name,
        parameters=list(op.parameters),
        precondition=precondition,
        effects=[eff],
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
    config: Optional[ScoringConfig] = None,
    mutex_groups: Optional[List[Any]] = None,
) -> DeltaMinimizationResult:
    """
    Iteratively synthesize deterministic recovery operators to make the
    abstract transition graph strongly connected (irreducible).

    Each recovery operator is computed purely from the predicate delta
    between a sink and source SCC — no LLM call.  The resulting
    ``ActionSchema`` has a single deterministic effect (prob 1.0) with
    no failure branch and no numeric rewards.

    Parameters
    ----------
    abstract_graph : nx.DiGraph
        The abstract transition graph from Step 2.
    domain : Domain
        The pyPPDDL Domain (will be mutated with new actions).
    config : ScoringConfig, optional
        Scoring parameters. Uses defaults if None.
    mutex_groups : list[ExactlyOneGroup], optional
        If provided, uses :func:`mutex_aware_hamming_distance` instead
        of the naïve logical Hamming distance.

    Returns
    -------
    DeltaMinimizationResult
    """
    if config is None:
        config = ScoringConfig()

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

                if mutex_groups:
                    delta = mutex_aware_hamming_distance(
                        sink_true, sink_false, src_true, src_false,
                        mutex_groups=mutex_groups,
                    )
                else:
                    delta = calculate_logical_hamming_distance(
                        sink_true, sink_false, src_true, src_false
                    )

                if delta > config.delta_threshold:
                    continue

                # Topological gain: prefer pairs where source can reach
                # sink in the DAG (adding sink→source closes a cycle).
                gain = 0
                if nx.has_path(condensation.dag, src_id, sink_id):
                    # This edge would close a cycle → high gain
                    # Estimate merged nodes by path length
                    try:
                        path_len = nx.shortest_path_length(
                            condensation.dag, src_id, sink_id
                        )
                        gain = path_len + 1  # nodes merged into SCC
                    except nx.NetworkXNoPath:
                        gain = 1
                else:
                    gain = 0  # no cycle formed, low priority

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

        # ── Pick the top candidate and synthesize deterministically ──
        candidate = candidates[0]
        logger.info(
            f"  Bridging: SCC-{candidate.sink_scc} → "
            f"SCC-{candidate.source_scc} (Δ={candidate.delta}, "
            f"gain={candidate.topological_gain}, "
            f"score={candidate.weighted_score:.3f})"
        )

        op = _synthesize_operator(candidate, domain)

        # Convert to ActionSchema and add to domain
        action_schema = _convert_to_action_schema(op)
        domain.actions.append(action_schema)
        operators.append(op)
        stats["successful"] += 1
        stats["deltas"].append(candidate.delta)

        # Add edge to working graph (connect representative members)
        sink_members = condensation.scc_state_map.get(candidate.sink_scc, [])
        src_members = condensation.scc_state_map.get(candidate.source_scc, [])
        if sink_members and src_members:
            from_node = sink_members[0]
            to_node = src_members[0]
            working_graph.add_edge(
                from_node,
                to_node,
                action=op.name,
                prob=1.0,
            )
            op.sink_node = from_node
            op.source_node = to_node

        logger.info(f"    ✓ Synthesized: {op.name} ({op.sink_node} → {op.source_node})")

        # Budget cap: stop after max_recovery_per_iter operators
        if (config.max_recovery_per_iter is not None
                and len(operators) >= config.max_recovery_per_iter):
            logger.info(
                f"  Budget cap reached ({config.max_recovery_per_iter} "
                f"operators this iteration)"
            )
            break

    # Final check
    final_condensation = condense_to_dag(working_graph)
    final_bound = compute_augmentation_bound(final_condensation)

    avg_delta = (
        sum(stats["deltas"]) / len(stats["deltas"]) if stats["deltas"] else 0
    )

    logger.info(
        f"Delta minimization complete: {len(operators)} operators synthesized, "
        f"avg delta={avg_delta:.1f}"
    )

    return DeltaMinimizationResult(
        operators=operators,
        stats=stats,
        final_sources=final_bound.sources,
        final_sinks=final_bound.sinks,
        is_irreducible=final_bound.is_already_irreducible,
        augmented_graph=working_graph,
    )
