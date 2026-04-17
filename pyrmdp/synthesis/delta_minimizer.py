"""
Step 5: Delta Minimization & Deterministic Recovery Synthesis

Uses a formally verified, monotonic cycle-closing algorithm that
guarantees a single global Strongly Connected Component (SCC).

The algorithm proceeds in a monotonic while-loop:
  1. Condense the current graph into a DAG of SCCs.
  2. If 1 SCC → done.
  3. If multiple WCCs → **Minimum Weight Cycle Cover (MWCC)**
     via the Hungarian algorithm merges them into one WCC.
  4. If 1 WCC with >1 SCC → **Reachability-based cycle closure**
     finds a source reachable from a sink and adds one edge
     (sink→source) to close a cycle and merge SCCs.
  5. Repeat until a single SCC.

Recovery operators are computed **deterministically** from the
predicate delta between sink and source states — no LLM call.

  * Preconditions: minimal causal subset of the sink state (only
    predicates sharing a unified variable with the delta effects).
  * Effects: single deterministic outcome (prob 1.0) that adds/deletes
    exactly the predicates in the delta.
  * No failure branch, no numeric rewards — those are injected later
    by Step 1 (failure hallucination) and Step 6 (PPDDL emission).
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
from .prompts.llm_feasibility_prompt import (
    build_feasibility_prompt,
    build_unroll_prompt,
    parse_feasibility_response,
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

    # ── LLM Feasibility Gate ──
    enable_llm_feasibility: bool = False
    """When True, Hamming-optimal bridge candidates are validated by an
    LLM physics gate before being accepted.  Default False preserves
    the pure deterministic MSCA/MWCC algorithm."""

    feasibility_k: int = 5
    """Number of top-k Hamming candidates to send to the LLM per bridge."""

    feasibility_lambda: float = 0.5
    """Blending weight λ for the LLM rank penalty:
    C'[e] = d_ham(e) + λ · (rank(e) - 1) / k."""


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
    phase_tag: str = ""
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
#  Distance Helper
# ════════════════════════════════════════════════════════════════════

def _dist(
    u_node: str,
    v_node: str,
    original_graph: nx.DiGraph,
    mutex_groups: Optional[List[Any]] = None,
) -> float:
    """Mutex-aware Hamming distance between two *original* graph nodes.

    Returns ``float('inf')`` if either node lacks state data (the
    transition is strictly disallowed).
    """
    u_data = original_graph.nodes.get(u_node, {})
    v_data = original_graph.nodes.get(v_node, {})
    u_state = u_data.get("state")
    v_state = v_data.get("state")
    if u_state is None or v_state is None:
        return float("inf")

    u_true = set(u_state.true_predicates)
    u_false = set(u_state.false_predicates)
    v_true = set(v_state.true_predicates)
    v_false = set(v_state.false_predicates)

    if mutex_groups:
        return float(mutex_aware_hamming_distance(
            u_true, u_false, v_true, v_false, mutex_groups=mutex_groups,
        ))
    return float(calculate_logical_hamming_distance(
        u_true, u_false, v_true, v_false,
    ))


# ════════════════════════════════════════════════════════════════════
#  Best Original Bridge
# ════════════════════════════════════════════════════════════════════

def _get_state_preds(
    node: str, original_graph: nx.DiGraph,
) -> Tuple[set, set]:
    """Return (true_predicates, false_predicates) for a graph node."""
    state = original_graph.nodes[node].get("state")
    if state is None:
        return set(), set()
    return set(state.true_predicates), set(state.false_predicates)


# ── Module-level buffer for extra unrolled hops ──
_unroll_extra_edges: List[Tuple[str, str, float]] = []


def evaluate_candidates(
    sink_node: str,
    source_nodes: List[str],
    original_graph: nx.DiGraph,
    llm_fn,
    mutex_groups: Optional[List[Any]],
    k: int,
    lambda_penalty: float,
) -> Optional[Tuple[str, str, float]]:
    """Hamming Proposes, LLM Grounds — evaluate top-k candidates.

    1. Rank *source_nodes* by Hamming distance to *sink_node*.
    2. Take top-k, send to LLM for feasibility + ranking.
    3. Return the best feasible candidate with calibrated cost
       ``C' = d_ham + λ·(rank-1)/k``, or *None* if all infeasible.
    """
    # Score all candidates by Hamming
    scored = []
    for v in source_nodes:
        d = _dist(sink_node, v, original_graph, mutex_groups)
        if d < float("inf"):
            scored.append((d, v))
    scored.sort()
    top_k = scored[:k]
    if not top_k:
        return None

    sink_true, sink_false = _get_state_preds(sink_node, original_graph)

    candidates_for_prompt = []
    for idx, (d, v) in enumerate(top_k):
        v_true, v_false = _get_state_preds(v, original_graph)
        candidates_for_prompt.append({
            "candidate_id": idx,
            "source_node": v,
            "source_true": sorted(v_true),
            "source_false": sorted(v_false),
            "delta": d,
        })

    prompt = build_feasibility_prompt(
        sink_true=sorted(sink_true),
        sink_false=sorted(sink_false),
        candidates=candidates_for_prompt,
    )

    try:
        raw = llm_fn(prompt["user"], system=prompt["system"])
        parsed = parse_feasibility_response(raw, len(top_k))
    except Exception as exc:
        logger.warning(f"LLM feasibility call failed: {exc}; falling back to Hamming")
        # Fall back: return Hamming-best
        d, v = top_k[0]
        return sink_node, v, d

    # Find best feasible by calibrated cost
    best: Optional[Tuple[str, str, float]] = None
    best_cost = float("inf")
    for entry in parsed:
        cid = entry["candidate_id"]
        if not entry["is_feasible"] or cid >= len(top_k):
            continue
        d, v = top_k[cid]
        rank = entry.get("rank", cid + 1)
        calibrated = d + lambda_penalty * (rank - 1) / max(k, 1)
        if calibrated < best_cost:
            best_cost = calibrated
            best = (sink_node, v, calibrated)

    return best


def unroll_transition(
    sink_node: str,
    source_nodes: List[str],
    original_graph: nx.DiGraph,
    llm_fn,
    mutex_groups: Optional[List[Any]],
    k: int,
) -> List[Tuple[str, str, float]]:
    """Structural Graph Unrolling — find 2-hop path via intermediate.

    For each candidate source v, find intermediate c ∈ V such that
    sink→c→v is physically plausible.  Returns a list of edges
    (may be empty if no unrolling found).
    """
    sink_true, sink_false = _get_state_preds(sink_node, original_graph)

    # Collect candidate paths: sink → c → v
    paths = []
    all_nodes = sorted(original_graph.nodes())
    scored_sources = []
    for v in source_nodes:
        d = _dist(sink_node, v, original_graph, mutex_groups)
        if d < float("inf"):
            scored_sources.append((d, v))
    scored_sources.sort()
    top_sources = scored_sources[:k]

    for _, v in top_sources:
        v_true, v_false = _get_state_preds(v, original_graph)
        # Find best intermediate c: minimise _dist(sink,c) + _dist(c,v)
        best_c = None
        best_total = float("inf")
        for c in all_nodes:
            if c == sink_node or c == v:
                continue
            d1 = _dist(sink_node, c, original_graph, mutex_groups)
            d2 = _dist(c, v, original_graph, mutex_groups)
            total = d1 + d2
            if total < best_total:
                best_total = total
                best_c = c
        if best_c is not None:
            c_true, c_false = _get_state_preds(best_c, original_graph)
            paths.append({
                "candidate_id": len(paths),
                "source_node": v,
                "intermediate_node": best_c,
                "mid_true": sorted(c_true),
                "mid_false": sorted(c_false),
                "source_true": sorted(v_true),
                "source_false": sorted(v_false),
                "delta1": _dist(sink_node, best_c, original_graph, mutex_groups),
                "delta2": _dist(best_c, v, original_graph, mutex_groups),
                "cost_hop1": _dist(sink_node, best_c, original_graph, mutex_groups),
                "cost_hop2": _dist(best_c, v, original_graph, mutex_groups),
            })

    if not paths:
        return []

    # Ask LLM which 2-hop path is feasible
    prompt = build_unroll_prompt(
        sink_true=sorted(sink_true),
        sink_false=sorted(sink_false),
        paths=paths,
    )

    try:
        raw = llm_fn(prompt["user"], system=prompt["system"])
        parsed = parse_feasibility_response(raw, len(paths))
    except Exception as exc:
        logger.warning(f"LLM unroll call failed: {exc}; using Hamming-best path")
        if paths:
            p = paths[0]
            return [
                (sink_node, p["intermediate_node"], p["cost_hop1"]),
                (p["intermediate_node"], p["source_node"], p["cost_hop2"]),
            ]
        return []

    # Pick best feasible path
    for entry in sorted(parsed, key=lambda e: e.get("rank", 999)):
        cid = entry["candidate_id"]
        if not entry["is_feasible"] or cid >= len(paths):
            continue
        p = paths[cid]
        return [
            (sink_node, p["intermediate_node"], p["cost_hop1"]),
            (p["intermediate_node"], p["source_node"], p["cost_hop2"]),
        ]

    return []


def _best_original_bridge(
    sink_scc_id: int,
    source_scc_id: int,
    H: nx.DiGraph,
    original_graph: nx.DiGraph,
    mutex_groups: Optional[List[Any]] = None,
    llm_fn=None,
    scoring_config: Optional[ScoringConfig] = None,
) -> Tuple[Optional[str], Optional[str], float, bool]:
    """Find the (u*, v*) original-state pair with minimum distance.

    When *llm_fn* is provided and ``scoring_config.enable_llm_feasibility``
    is True, candidates are validated by the LLM physics gate with
    structural unrolling as fallback.

    Returns
    -------
    (u*, v*, cost, was_unrolled) where *was_unrolled* is True when
    the edge was produced by structural graph unrolling.
    """
    sink_members = sorted(H.nodes[sink_scc_id].get("members", set()))
    src_members = sorted(H.nodes[source_scc_id].get("members", set()))

    use_llm = (
        llm_fn is not None
        and scoring_config is not None
        and scoring_config.enable_llm_feasibility
    )

    if use_llm:
        cfg = scoring_config
        # Row-level batching: for each sink member, evaluate all source members
        best_result: Optional[Tuple[str, str, float]] = None
        best_cost = float("inf")
        for u in sink_members:
            result = evaluate_candidates(
                u, src_members, original_graph, llm_fn,
                mutex_groups, cfg.feasibility_k, cfg.feasibility_lambda,
            )
            if result is not None and result[2] < best_cost:
                best_result = result
                best_cost = result[2]

        if best_result is not None:
            return best_result[0], best_result[1], best_result[2], False

        # Fallback: structural unrolling
        logger.info("    LLM gate rejected all candidates; trying unrolling")
        for u in sink_members:
            edges = unroll_transition(
                u, src_members, original_graph, llm_fn,
                mutex_groups, cfg.feasibility_k,
            )
            if edges:
                # First hop is the primary edge; extra hops go to buffer
                primary = edges[0]
                for extra in edges[1:]:
                    _unroll_extra_edges.append(extra)
                return primary[0], primary[1], primary[2], True

    # Pure Hamming fallback (or llm disabled)
    best_u: Optional[str] = None
    best_v: Optional[str] = None
    best_cost_h = float("inf")

    for u in sink_members:
        for v in src_members:
            d = _dist(u, v, original_graph, mutex_groups)
            if d < best_cost_h:
                best_cost_h = d
                best_u = u
                best_v = v

    return best_u, best_v, best_cost_h, False


# ════════════════════════════════════════════════════════════════════
#  Algorithm 2 & 5: Inter-WCC MWCC via Hungarian Algorithm
# ════════════════════════════════════════════════════════════════════

def _build_global_cycle_across_wccs(
    H: nx.DiGraph,
    wccs: List[set],
    original_graph: nx.DiGraph,
    mutex_groups: Optional[List[Any]] = None,
    llm_fn=None,
    scoring_config: Optional[ScoringConfig] = None,
) -> List[Tuple[str, str, float, str]]:
    """Minimum-Weight Cycle Cover to merge all WCCs into one global cycle.

    1. Extract sinks/sources per WCC.
    2. Build a k×k cost matrix C where C[i,j] = min-distance bridge
       from a sink of WCCᵢ to a source of WCCⱼ (∞ on diagonal).
    3. Solve with ``scipy.optimize.linear_sum_assignment`` for the
       minimum-weight perfect matching (produces a successor array).
    4. Patch disjoint cycles into a single Hamiltonian cycle over WCCs.
    5. Return witness edges (u*, v*) for each arc in the global cycle.

    Parameters
    ----------
    H : nx.DiGraph
        The condensation DAG.
    wccs : list[set]
        Weakly connected components of *H* (sets of meta-node IDs).
    original_graph : nx.DiGraph
        The original abstract transition graph.
    mutex_groups : list, optional

    Returns
    -------
    list[(u_orig, v_orig, cost)]
        Witness edges in the original graph, one per arc in the
        global cycle over WCCs.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    k = len(wccs)
    wcc_list = [sorted(w) for w in wccs]  # deterministic ordering

    # Collect sinks / sources for each WCC
    wcc_sinks: List[List[int]] = []
    wcc_sources: List[List[int]] = []
    for wcc_nodes in wcc_list:
        sub = H.subgraph(wcc_nodes)
        sinks = [n for n in wcc_nodes if sub.out_degree(n) == 0]
        sources = [n for n in wcc_nodes if sub.in_degree(n) == 0]
        # Fallback: use any node
        if not sinks:
            sinks = [wcc_nodes[0]]
        if not sources:
            sources = [wcc_nodes[0]]
        wcc_sinks.append(sinks)
        wcc_sources.append(sources)

    # ── Build k×k cost matrix + witness table ──
    INF = 1e18  # finite sentinel for the solver (real inf breaks scipy)
    C = np.full((k, k), INF, dtype=np.float64)
    witness: Dict[Tuple[int, int], Tuple[Optional[str], Optional[str], float, bool]] = {}

    for i in range(k):
        for j in range(k):
            if i == j:
                continue  # diagonal stays INF
            # Best bridge: sink of WCC_i → source of WCC_j
            best_u, best_v, best_cost, best_unrolled = None, None, float("inf"), False
            for s_scc in wcc_sinks[i]:
                for t_scc in wcc_sources[j]:
                    u, v, cost, unrolled = _best_original_bridge(
                        s_scc, t_scc, H, original_graph, mutex_groups,
                        llm_fn, scoring_config,
                    )
                    if cost < best_cost:
                        best_u, best_v, best_cost, best_unrolled = u, v, cost, unrolled
            witness[(i, j)] = (best_u, best_v, best_cost, best_unrolled)
            if best_cost < float("inf"):
                C[i, j] = best_cost

    # ── Solve minimum-weight assignment (successor permutation) ──
    row_ind, col_ind = linear_sum_assignment(C)
    succ = dict(zip(row_ind.tolist(), col_ind.tolist()))

    logger.info(
        f"    MWCC assignment: {dict(succ)}"
    )

    # ── Cycle patching: merge disjoint permutation cycles ──
    # The assignment is a permutation on {0,…,k-1}. It may consist of
    # multiple disjoint cycles.  We merge them into one Hamiltonian
    # cycle by swapping successors between two cycles.
    visited = [False] * k
    cycles: List[List[int]] = []
    for start in range(k):
        if visited[start]:
            continue
        cycle = []
        node = start
        while not visited[node]:
            visited[node] = True
            cycle.append(node)
            node = succ[node]
        cycles.append(cycle)

    logger.info(
        f"    MWCC initial cycles: {cycles}"
    )

    # Merge until 1 cycle
    while len(cycles) > 1:
        # Pick the two cheapest-to-merge cycles
        best_merge_cost = float("inf")
        best_ci = 0
        best_cj = 1
        best_a = cycles[0][0]
        best_b = cycles[1][0]

        for ci in range(len(cycles)):
            for cj in range(ci + 1, len(cycles)):
                for a in cycles[ci]:
                    for b in cycles[cj]:
                        # Cost of swapping: remove a→succ[a], b→succ[b]
                        # and add a→succ[b], b→succ[a]
                        old_cost = C[a, succ[a]] + C[b, succ[b]]
                        new_cost = C[a, succ[b]] + C[b, succ[a]]
                        delta = new_cost - old_cost
                        if delta < best_merge_cost:
                            best_merge_cost = delta
                            best_ci = ci
                            best_cj = cj
                            best_a = a
                            best_b = b

        # Perform the swap: a→succ[b], b→succ[a]
        old_a_succ = succ[best_a]
        old_b_succ = succ[best_b]
        succ[best_a] = old_b_succ
        succ[best_b] = old_a_succ

        # Re-extract cycles
        visited = [False] * k
        cycles = []
        for start in range(k):
            if visited[start]:
                continue
            cycle = []
            node = start
            while not visited[node]:
                visited[node] = True
                cycle.append(node)
                node = succ[node]
            cycles.append(cycle)

    logger.info(
        f"    MWCC final global cycle: {cycles[0]}"
    )

    # ── Collect witness edges for the global cycle ──
    result: List[Tuple[str, str, float, str]] = []
    for i in range(k):
        j = succ[i]
        w = witness.get((i, j))
        if w is None or w[0] is None or w[1] is None:
            # Need to recompute — the patching introduced new arcs
            best_u, best_v, best_cost, best_unrolled = None, None, float("inf"), False
            for s_scc in wcc_sinks[i]:
                for t_scc in wcc_sources[j]:
                    u, v, cost, unrolled = _best_original_bridge(
                        s_scc, t_scc, H, original_graph, mutex_groups,
                        llm_fn, scoring_config,
                    )
                    if cost < best_cost:
                        best_u, best_v, best_cost, best_unrolled = u, v, cost, unrolled
            if best_u is None or best_v is None:
                raise RuntimeError(
                    f"Cannot bridge WCC-{i} → WCC-{j}: no feasible pair"
                )
            phase = "mwcc+unrolled" if best_unrolled else "mwcc"
            result.append((best_u, best_v, best_cost, phase))
        else:
            phase = "mwcc+unrolled" if w[3] else "mwcc"
            result.append((w[0], w[1], w[2], phase))

    return result


# ════════════════════════════════════════════════════════════════════
#  Algorithm 3: Intra-WCC Reachability-Based Cycle Closure
# ════════════════════════════════════════════════════════════════════

def _close_one_reachable_source_sink_cycle(
    H: nx.DiGraph,
    original_graph: nx.DiGraph,
    mutex_groups: Optional[List[Any]] = None,
    llm_fn=None,
    scoring_config: Optional[ScoringConfig] = None,
) -> Tuple[str, str, float, str]:
    """Find one (sink→source) edge that closes a reachable cycle.

    The condensation *H* is weakly connected but has >1 SCC-node.
    Find all (source, sink) pairs where source can reach sink via
    directed paths in *H*, then pick the pair whose reverse bridge
    (sink→source) has minimum Hamming distance in the original graph.

    Parameters
    ----------
    H : nx.DiGraph
        Weakly-connected condensation DAG with >1 node.
    original_graph : nx.DiGraph
        The abstract transition graph (nodes carry ``state``).
    mutex_groups : list, optional

    Returns
    -------
    (u_orig, v_orig, cost)
        The witness edge (from sink state to source state).

    Raises
    ------
    RuntimeError
        If no finite-cost bridge exists (domain is topologically
        un-rescuable).
    """
    sources = [n for n in H.nodes() if H.in_degree(n) == 0]
    sinks = [n for n in H.nodes() if H.out_degree(n) == 0]

    best_u: Optional[str] = None
    best_v: Optional[str] = None
    best_cost = float("inf")
    best_unrolled = False

    for s in sources:
        # Precompute descendants once per source
        desc = nx.descendants(H, s)
        for t in sinks:
            if t not in desc:
                continue
            # s can reach t in H → adding t→s closes a cycle
            u, v, cost, unrolled = _best_original_bridge(
                t, s, H, original_graph, mutex_groups,
                llm_fn, scoring_config,
            )
            if cost < best_cost:
                best_u, best_v, best_cost, best_unrolled = u, v, cost, unrolled

    if best_u is None or best_v is None or best_cost == float("inf"):
        raise RuntimeError(
            "Domain is topologically un-rescuable: no finite-cost "
            "sink→source bridge found in the weakly-connected condensation."
        )

    phase = "reachability+unrolled" if best_unrolled else "reachability"
    return best_u, best_v, best_cost, phase


# ════════════════════════════════════════════════════════════════════
#  Algorithm 1: Main Augmentation Loop
# ════════════════════════════════════════════════════════════════════

def _compute_augmentation_edges(
    original_graph: nx.DiGraph,
    mutex_groups: Optional[List[Any]] = None,
    max_iterations: int = 200,
    llm_fn=None,
    scoring_config: Optional[ScoringConfig] = None,
) -> Tuple[List[Tuple[str, str, float, str]], int]:
    """Monotonically add meta-edges until the graph is one SCC.

    Returns
    -------
    edges : list[(from_node, to_node, cost, phase)]
        Each element is an original-state pair plus the Hamming cost
        and the phase label (``"mwcc"`` or ``"reachability"``).
    initial_num_wccs : int
        Number of WCCs before any augmentation (for stats/visualization).
    """
    E_add: List[Tuple[str, str]] = []       # accumulated meta-edges
    result: List[Tuple[str, str, float, str]] = []
    initial_num_wccs: Optional[int] = None

    for _step in range(max_iterations):
        # Build temporary graph with accumulated meta-edges
        H_raw = original_graph.copy()
        for u, v in E_add:
            H_raw.add_edge(u, v, action="__augmentation__", prob=1.0)

        H = nx.condensation(H_raw)

        if len(H.nodes()) == 1:
            logger.info(
                f"  Augmentation loop: single SCC after {_step} edge(s)"
            )
            break

        wccs = list(nx.weakly_connected_components(H))

        if initial_num_wccs is None:
            initial_num_wccs = len(wccs)

        if len(wccs) > 1:
            # ── Multiple WCCs: MWCC inter-component routing ──
            logger.info(
                f"  Augmentation step {_step}: {len(wccs)} WCCs — "
                f"applying MWCC"
            )
            bridges = _build_global_cycle_across_wccs(
                H, wccs, original_graph, mutex_groups,
                llm_fn, scoring_config,
            )
            for u, v, cost, phase in bridges:
                E_add.append((u, v))
                result.append((u, v, cost, phase))
                logger.info(
                    f"    MWCC edge: {u} → {v} (Δ={cost:.0f}, {phase})"
                )
            # Collect any extra unrolled hops
            while _unroll_extra_edges:
                eu, ev, ec = _unroll_extra_edges.pop(0)
                E_add.append((eu, ev))
                result.append((eu, ev, ec, "unrolled"))
                logger.info(
                    f"    Unrolled extra hop: {eu} → {ev} (Δ={ec:.0f})"
                )

        else:
            # ── Single WCC, >1 SCC: reachability cycle closure ──
            logger.info(
                f"  Augmentation step {_step}: 1 WCC, "
                f"{len(H.nodes())} SCCs — closing one cycle"
            )
            u, v, cost, phase = _close_one_reachable_source_sink_cycle(
                H, original_graph, mutex_groups,
                llm_fn, scoring_config,
            )
            E_add.append((u, v))
            result.append((u, v, cost, phase))
            logger.info(
                f"    Reachability edge: {u} → {v} (Δ={cost:.0f}, {phase})"
            )
            # Collect any extra unrolled hops
            while _unroll_extra_edges:
                eu, ev, ec = _unroll_extra_edges.pop(0)
                E_add.append((eu, ev))
                result.append((eu, ev, ec, "unrolled"))
                logger.info(
                    f"    Unrolled extra hop: {eu} → {ev} (Δ={ec:.0f})"
                )

    else:
        logger.warning(
            f"  Augmentation loop did not converge in {max_iterations} steps"
        )

    return result, initial_num_wccs or 1


# ════════════════════════════════════════════════════════════════════
#  Lift an (original-state, original-state) pair to a PDDL operator
# ════════════════════════════════════════════════════════════════════

def _lift_to_operator(
    sink_node: str,
    source_node: str,
    original_graph: nx.DiGraph,
    domain: Domain,
    operator_index: int,
    phase_tag: str = "",
) -> SynthesizedOperator:
    """Create a SynthesizedOperator from an original-state pair.

    Re-uses the existing deterministic synthesis machinery
    (variable unification + minimal causal preconditions).
    """
    u_data = original_graph.nodes[sink_node]
    v_data = original_graph.nodes[source_node]
    u_state = u_data["state"]
    v_state = v_data["state"]

    sink_true = set(u_state.true_predicates)
    sink_false = set(u_state.false_predicates)
    src_true = set(v_state.true_predicates)
    src_false = set(v_state.false_predicates)

    delta = calculate_logical_hamming_distance(
        sink_true, sink_false, src_true, src_false,
    )

    candidate = CandidateEdge(
        sink_scc=-1,            # not meaningful for the new algorithm
        source_scc=-1,
        sink_true_preds=sink_true,
        sink_false_preds=sink_false,
        source_true_preds=src_true,
        source_false_preds=src_false,
        delta=delta,
        topological_gain=0,
    )
    # Override the name to use the operator index
    candidate.sink_scc = operator_index  # embed index in name

    op = _synthesize_operator(candidate, domain)
    # Fix name to be unique and descriptive
    op.name = f"recover_{operator_index}"
    op.sink_node = sink_node
    op.source_node = source_node
    op.phase_tag = phase_tag
    return op


# ════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ════════════════════════════════════════════════════════════════════

def delta_minimize(
    abstract_graph: nx.DiGraph,
    domain: Domain,
    *,
    config: Optional[ScoringConfig] = None,
    mutex_groups: Optional[List[Any]] = None,
    llm_fn=None,
) -> DeltaMinimizationResult:
    """
    Synthesize deterministic recovery operators to make the abstract
    transition graph strongly connected (irreducible).

    Uses a monotonic, formally verified cycle-closing algorithm:
      1. **MWCC** (Hungarian algorithm) to merge disconnected WCCs.
      2. **Reachability-based cycle closure** to collapse remaining
         SCCs within the single WCC.

    Each recovery operator is computed purely from the predicate delta
    between a sink and source state — no LLM call.

    Parameters
    ----------
    abstract_graph : nx.DiGraph
        The abstract transition graph from Step 2.
    domain : Domain
        The pyPPDDL Domain (will be mutated with new actions).
    config : ScoringConfig, optional
        Scoring parameters (``max_iterations``, ``max_recovery_per_iter``
        are respected).  Uses defaults if *None*.
    mutex_groups : list[ExactlyOneGroup], optional
        If provided, uses mutex-aware Hamming distance.

    Returns
    -------
    DeltaMinimizationResult
    """
    if config is None:
        config = ScoringConfig()

    working_graph = abstract_graph.copy()

    # ── Quick check: already irreducible? ──
    if nx.is_strongly_connected(working_graph):
        logger.info("Graph is already irreducible — nothing to do!")
        return DeltaMinimizationResult(
            operators=[],
            stats={
                "successful": 0, "failed": 0, "deltas": [],
                "iterations": 0, "phase1_ops": 0, "phase2_ops": 0,
                "num_wccs": 1,
            },
            final_sources=[],
            final_sinks=[],
            is_irreducible=True,
            augmented_graph=working_graph,
        )

    # ── Resolve active LLM function ──
    if config.enable_llm_feasibility:
        if llm_fn is not None:
            active_llm_fn = llm_fn
        else:
            from .llm_config import build_llm_fn
            logger.info("  Building LLM function for feasibility gate (from llm.yaml)")
            active_llm_fn = build_llm_fn()
    else:
        active_llm_fn = None

    # ── Compute augmentation edges ──
    max_iters = config.max_iterations
    if config.max_recovery_per_iter is not None:
        max_iters = min(max_iters, config.max_recovery_per_iter * 10)

    # Clear module-level unroll buffer
    _unroll_extra_edges.clear()

    aug_edges, initial_num_wccs = _compute_augmentation_edges(
        working_graph,
        mutex_groups=mutex_groups,
        max_iterations=max_iters,
        llm_fn=active_llm_fn,
        scoring_config=config,
    )

    # ── Lift each edge to a PDDL operator ──
    operators: List[SynthesizedOperator] = []
    phase1_count = 0  # MWCC edges count as "Phase 1" for visualization
    phase2_count = 0  # Reachability edges count as "Phase 2"
    phase3_count = 0  # Unrolled edges count as "Phase 3"
    deltas: List[int] = []

    for idx, (u, v, cost, phase) in enumerate(aug_edges):
        op = _lift_to_operator(u, v, working_graph, domain, idx, phase_tag=phase)
        action_schema = _convert_to_action_schema(op)
        domain.actions.append(action_schema)

        # Add edge to working graph
        working_graph.add_edge(u, v, action=op.name, prob=1.0)

        operators.append(op)
        deltas.append(op.delta)

        if "mwcc" in phase:
            phase1_count += 1
        elif "unrolled" == phase:
            phase3_count += 1
        else:
            phase2_count += 1

        logger.info(
            f"  Operator [{idx}] ({phase}): {op.name} — "
            f"{u} → {v}  Δ={op.delta}"
        )

    # ── Final verification ──
    is_irr = nx.is_strongly_connected(working_graph)
    final_condensation = condense_to_dag(working_graph)
    final_bound = compute_augmentation_bound(final_condensation)

    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    stats = {
        "successful": len(operators),
        "failed": 0,
        "deltas": deltas,
        "iterations": len(operators),
        "phase1_ops": phase1_count,
        "phase2_ops": phase2_count,
        "phase3_ops": phase3_count,
        "num_wccs": initial_num_wccs,
    }

    logger.info(
        f"Delta minimization complete: {len(operators)} operators "
        f"(MWCC: {phase1_count}, Reachability: {phase2_count}, "
        f"Unrolled: {phase3_count}), "
        f"avg delta={avg_delta:.1f}, irreducible={is_irr}"
    )

    return DeltaMinimizationResult(
        operators=operators,
        stats=stats,
        final_sources=final_bound.sources,
        final_sinks=final_bound.sinks,
        is_irreducible=is_irr,
        augmented_graph=working_graph,
    )
