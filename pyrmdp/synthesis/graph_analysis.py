"""
Steps 3 & 4: Graph Condensation (SCC → DAG) and Augmentation Bound (MSCA)

Step 3: Extract Strongly Connected Components of the abstract transition graph
        and condense into a DAG.
Step 4: Identify topological sources/sinks and compute the minimum number of
        new transitions needed for irreducibility (MSCA theorem).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Step 3: SCC Condensation
# ════════════════════════════════════════════════════════════════════

@dataclass
class CondensationResult:
    """Result of condensing the abstract transition graph into a DAG."""
    dag: nx.DiGraph
    scc_state_map: Dict[int, List[str]]  # SCC node ID → list of original node IDs
    scc_label_map: Dict[int, str]        # SCC node ID → human-readable label
    original_graph: nx.DiGraph
    num_sccs: int
    is_strongly_connected: bool


def condense_to_dag(
    abstract_graph: nx.DiGraph,
) -> CondensationResult:
    """
    Extract the Strongly Connected Components of the abstract transition
    graph and condense them into a DAG.

    Each node in the DAG represents an SCC — a set of abstract states
    that can already reach each other via existing transitions.

    Parameters
    ----------
    abstract_graph : nx.DiGraph
        The abstract transition graph from fodd_builder.enumerate_abstract_states().

    Returns
    -------
    CondensationResult
        Contains the DAG, SCC-to-state mappings, and connectivity info.
    """
    # Check if already strongly connected
    is_sc = nx.is_strongly_connected(abstract_graph)

    # Compute condensation
    dag = nx.condensation(abstract_graph)
    num_sccs = dag.number_of_nodes()

    # Build SCC → original node mapping
    # nx.condensation stores the 'members' attribute on each DAG node
    scc_state_map: Dict[int, List[str]] = {}
    scc_label_map: Dict[int, str] = {}

    for scc_id in dag.nodes():
        members = sorted(dag.nodes[scc_id].get("members", set()))
        scc_state_map[scc_id] = members

        # Build a label from the member states
        if len(members) <= 3:
            member_labels = []
            for m in members:
                if "label" in abstract_graph.nodes.get(m, {}):
                    member_labels.append(abstract_graph.nodes[m]["label"])
                else:
                    member_labels.append(str(m))
            scc_label_map[scc_id] = f"SCC-{scc_id}: {', '.join(member_labels)}"
        else:
            scc_label_map[scc_id] = f"SCC-{scc_id} ({len(members)} states)"

    logger.info(
        f"Condensation: {abstract_graph.number_of_nodes()} states → "
        f"{num_sccs} SCCs (strongly connected: {is_sc})"
    )

    return CondensationResult(
        dag=dag,
        scc_state_map=scc_state_map,
        scc_label_map=scc_label_map,
        original_graph=abstract_graph,
        num_sccs=num_sccs,
        is_strongly_connected=is_sc,
    )


# ════════════════════════════════════════════════════════════════════
#  Step 4: MSCA Augmentation Bound
# ════════════════════════════════════════════════════════════════════

@dataclass
class AugmentationBound:
    """Result of the MSCA augmentation bound calculation."""
    sources: List[int]       # SCC node IDs with in-degree = 0
    sinks: List[int]         # SCC node IDs with out-degree = 0
    bound: int               # max(|sources|, |sinks|)
    is_already_irreducible: bool


def compute_augmentation_bound(
    condensation: CondensationResult,
) -> AugmentationBound:
    """
    Identify topological sources (in-degree = 0) and sinks (out-degree = 0)
    in the condensation DAG.

    The theoretical minimum number of new edges required to make the original
    graph strongly connected is max(|sources|, |sinks|), per the Minimum
    Strong Connectivity Augmentation (MSCA) theorem.

    Special case: if the DAG has exactly 1 node (already strongly connected),
    the bound is 0.

    Parameters
    ----------
    condensation : CondensationResult
        Output from condense_to_dag().

    Returns
    -------
    AugmentationBound
    """
    dag = condensation.dag

    # Special case: single SCC = already irreducible
    if condensation.num_sccs <= 1:
        return AugmentationBound(
            sources=[], sinks=[], bound=0, is_already_irreducible=True
        )

    # Compute in/out degrees
    in_deg = dict(dag.in_degree())
    out_deg = dict(dag.out_degree())

    sources = [node for node, deg in in_deg.items() if deg == 0]
    sinks = [node for node, deg in out_deg.items() if deg == 0]

    bound = max(len(sources), len(sinks))

    logger.info(
        f"MSCA bound: sources={len(sources)}, sinks={len(sinks)}, "
        f"bound={bound}"
    )
    logger.info(f"  Sources: {sources}")
    logger.info(f"  Sinks:   {sinks}")

    return AugmentationBound(
        sources=sources,
        sinks=sinks,
        bound=bound,
        is_already_irreducible=False,
    )


# ════════════════════════════════════════════════════════════════════
#  Helper: get representative state for an SCC
# ════════════════════════════════════════════════════════════════════

def get_scc_representative_predicates(
    scc_id: int,
    condensation: CondensationResult,
) -> Tuple[Set[str], Set[str]]:
    """
    Get the (true_predicates, false_predicates) of a representative state
    from the given SCC.

    Uses the first member state's AbstractState attributes.
    """
    members = condensation.scc_state_map.get(scc_id, [])
    if not members:
        return set(), set()

    first_member = members[0]
    node_data = condensation.original_graph.nodes.get(first_member, {})
    state = node_data.get("state")

    if state is None:
        return set(), set()

    return set(state.true_predicates), set(state.false_predicates)
