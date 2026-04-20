"""Graph intersection & recall metrics.

Loads the pyrmdp abstract graph and evaluates how well an empirical
lifted trajectory is covered by the graph's states and transitions.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

LiftedState = Tuple[FrozenSet[str], FrozenSet[str]]


# ─────────────────────────────────────────────────────────────
#  Graph loading
# ─────────────────────────────────────────────────────────────

def load_abstract_graph(pipeline_dir: str | Path) -> nx.DiGraph:
    """Load the final iteration's abstract graph from GraphML."""
    d = Path(pipeline_dir)
    candidates = sorted(d.glob("iter*_step2_abstract_graph.graphml"))
    if not candidates:
        raise FileNotFoundError(f"No abstract graph found in {d}")
    path = candidates[-1]
    G = nx.read_graphml(path)
    logger.info(
        "Loaded abstract graph from %s (%d nodes, %d edges)",
        path.name, G.number_of_nodes(), G.number_of_edges(),
    )
    return G


def _parse_frozenset(text: str) -> FrozenSet[str]:
    """Parse ``frozenset({'a', 'b'})`` or ``frozenset()`` from string."""
    m = re.search(r"frozenset\(\{([^}]*)\}\)", text)
    if m:
        inner = m.group(1)
        items = [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
        return frozenset(items)
    if "frozenset()" in text:
        return frozenset()
    return frozenset()


def parse_graph_nodes(
    G: nx.DiGraph,
) -> Dict[LiftedState, str]:
    """Extract ``(true_preds, false_preds) → node_id`` mapping.

    Parses the ``d0`` attribute which stores the ``AbstractState`` repr.
    """
    mapping: Dict[LiftedState, str] = {}

    for nid, data in G.nodes(data=True):
        raw = data.get("state", data.get("d0", ""))
        if "true_predicates=" not in raw:
            continue

        true_preds = _parse_frozenset(
            raw.split("true_predicates=")[1].split(", false_predicates=")[0]
        )
        false_part = raw.split("false_predicates=")[1]
        false_preds = _parse_frozenset(false_part)

        key = (true_preds, false_preds)
        mapping[key] = nid

    logger.info("Parsed %d graph nodes with state signatures", len(mapping))
    return mapping


def graph_predicate_vocabulary(G: nx.DiGraph) -> FrozenSet[str]:
    """Return the set of bare predicate names that appear in graph nodes."""
    nodes = parse_graph_nodes(G)
    vocab: set = set()
    for tp, fp in nodes:
        vocab |= tp | fp
    return frozenset(vocab)


# ─────────────────────────────────────────────────────────────
#  Node matching
# ─────────────────────────────────────────────────────────────

def find_matching_nodes(
    lifted_state: LiftedState,
    graph_nodes: Dict[LiftedState, str],
) -> List[str]:
    """Find all graph nodes compatible with a partial empirical observation.

    Empirical states use existential abstraction (∃):
      - emp_true predicates  ⟹  ∃x P(x) observed True
      - emp_false predicates ⟹  ¬∃x P(x), all groundings observed False

    A graph node's vocabulary is the union of its true and false predicate
    sets.  Only predicates that the node actually tracks are used for the
    subset check — predicates the VLM observes but the node ignores are
    skipped.  This handles the fact that both the VLM observation and the
    graph node may be partial views of the full state.

    Returns all matching node IDs (the belief state under partial
    observability).
    """
    emp_true, emp_false = lifted_state
    matches: List[str] = []
    for (g_true, g_false), nid in graph_nodes.items():
        node_vocab = g_true | g_false
        relevant_emp_true = emp_true & node_vocab
        relevant_emp_false = emp_false & node_vocab
        if relevant_emp_true.issubset(g_true) and relevant_emp_false.issubset(g_false):
            matches.append(nid)
    return matches


# ─────────────────────────────────────────────────────────────
#  Evaluation metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    state_recall: float = 0.0
    path_recall: float = 0.0
    super_coverage: float = 0.0
    total_keyframes: int = 0
    matched_keyframes: int = 0
    total_transitions: int = 0
    matched_transitions: int = 0
    unique_empirical_states: int = 0
    total_graph_nodes: int = 0
    belief_states: List[List[str]] = field(default_factory=list)
    unmatched_states: List[Dict] = field(default_factory=list)


def evaluate(
    lifted_trajectory: List[LiftedState],
    G: nx.DiGraph,
) -> EvalResult:
    """Compute State Recall, Path Recall, and Super-Coverage.

    Uses **subset matching**: each empirical keyframe is a partial
    observation that maps to a *Belief State* (set of compatible graph
    nodes).  A keyframe is "recalled" if its belief state is non-empty.
    A transition is "recalled" if any pair ``(u, v)`` across consecutive
    belief states is reachable in the graph.
    """
    if not lifted_trajectory:
        return EvalResult()

    graph_nodes = parse_graph_nodes(G)
    result = EvalResult(
        total_keyframes=len(lifted_trajectory),
        total_graph_nodes=G.number_of_nodes(),
    )

    beliefs: List[List[str]] = []
    for ls in lifted_trajectory:
        matched = find_matching_nodes(ls, graph_nodes)
        beliefs.append(matched)
        if matched:
            result.matched_keyframes += 1
        else:
            result.unmatched_states.append({
                "true": sorted(ls[0]),
                "false": sorted(ls[1]),
            })

    result.belief_states = beliefs
    result.state_recall = (
        result.matched_keyframes / result.total_keyframes
        if result.total_keyframes > 0 else 0.0
    )

    if len(lifted_trajectory) >= 2:
        result.total_transitions = len(lifted_trajectory) - 1
        matched_paths = 0
        for i in range(result.total_transitions):
            b_src, b_tgt = beliefs[i], beliefs[i + 1]
            if not b_src or not b_tgt:
                continue
            found = False
            for u in b_src:
                if found:
                    break
                for v in b_tgt:
                    if u == v:
                        found = True
                        break
                    if G.has_node(u) and G.has_node(v):
                        try:
                            if nx.has_path(G, u, v):
                                found = True
                                break
                        except nx.NetworkXError:
                            pass
            if found:
                matched_paths += 1
        result.matched_transitions = matched_paths
        result.path_recall = (
            matched_paths / result.total_transitions
            if result.total_transitions > 0 else 0.0
        )

    unique = set(lifted_trajectory)
    result.unique_empirical_states = len(unique)
    result.super_coverage = (
        result.total_graph_nodes / result.unique_empirical_states
        if result.unique_empirical_states > 0 else 0.0
    )

    logger.info(
        "Evaluation: State Recall=%.2f (%d/%d), Path Recall=%.2f (%d/%d), "
        "Super-Coverage=%.2f (%d/%d)",
        result.state_recall, result.matched_keyframes, result.total_keyframes,
        result.path_recall, result.matched_transitions, result.total_transitions,
        result.super_coverage, result.total_graph_nodes, result.unique_empirical_states,
    )
    return result
