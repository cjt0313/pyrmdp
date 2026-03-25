"""
Step 2: Abstract State Space Extraction via Lifted FODDs

Bridges pyPPDDL's parsed ActionSchema objects into pyrmdp's FODDManager
to build lifted transition FODDs. Each action's precondition formula
(AND/OR/NOT tree) is compiled into a FODD where internal nodes are Atom
tests and leaves encode effect outcomes.

FODD leaf-paths then define abstract state partitions (the quotient graph),
where each path corresponds to a distinct precondition context.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from ..core.fodd import FODDManager, FODDNode
from ..core.logic import Atom, Constant, Variable
from ..pruning.reduction import SyntacticReducer, apply

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
#  Data Model Bridge: pyPPDDL → pyrmdp
# ════════════════════════════════════════════════════════════════════

def _term_to_pyrmdp(token: str) -> Variable | Constant:
    """Convert a PDDL token to a pyrmdp Variable or Constant."""
    if isinstance(token, str) and token.startswith("?"):
        return Variable(name=token)
    return Constant(name=str(token))


def map_pyppddl_to_pyrmdp(pred_tuple: tuple) -> Atom:
    """
    Convert a pyPPDDL predicate tuple (pred_name, arg1, arg2, ...)
    into a pyrmdp Atom.

    Example
    -------
    >>> map_pyppddl_to_pyrmdp(("arm-at", "?loc"))
    Atom(predicate="arm-at", terms=[Variable(name="?loc")])
    """
    predicate_name = pred_tuple[0]
    terms = [_term_to_pyrmdp(t) for t in pred_tuple[1:]]
    return Atom(predicate=predicate_name, terms=terms)


def _predicate_schema_to_atom(pred: Predicate) -> Atom:
    """Convert a pyPPDDL Predicate schema definition to a pyrmdp Atom."""
    terms = [Variable(name=p.name) for p in pred.parameters]
    return Atom(predicate=pred.name, terms=terms)


def build_global_order(domain: Domain) -> List[str]:
    """
    Build a global predicate ordering from the domain definition.
    Predicates are ordered by their position in the :predicates block.
    """
    return [p.name for p in domain.predicates]


# ════════════════════════════════════════════════════════════════════
#  Precondition Tree → FODD
# ════════════════════════════════════════════════════════════════════

def _collect_atoms_from_precondition(precond: Any) -> List[Atom]:
    """
    Recursively walk a pyPPDDL precondition S-expression tree and
    collect all Atom objects (ignoring and/or/not structure).
    """
    if precond is None:
        return []
    if not isinstance(precond, list) or len(precond) == 0:
        return []

    head = precond[0]

    if head in ("and", "or"):
        atoms = []
        for child in precond[1:]:
            atoms.extend(_collect_atoms_from_precondition(child))
        return atoms

    if head == "not":
        return _collect_atoms_from_precondition(precond[1])

    # Numeric comparison — skip, not a predicate atom
    if head in (">", "<", ">=", "<=", "="):
        return []

    # It's a predicate reference: (pred_name ?arg1 ?arg2 ...)
    atom = map_pyppddl_to_pyrmdp(tuple(precond))
    return [atom]


def build_precondition_fodd(
    action: ActionSchema,
    manager: FODDManager,
    val_applicable: int,
    val_not_applicable: int,
) -> int:
    """
    Compile an action's precondition S-expression tree into a FODD.

    Internal nodes test precondition atoms in the manager's global order.
    The high (True) branch ultimately leads to *val_applicable*; the low
    (False) branch leads to *val_not_applicable*.

    Handles AND, OR, NOT, and nested combinations.

    Parameters
    ----------
    action : ActionSchema
        The pyPPDDL action with a .precondition S-expression.
    manager : FODDManager
        The FODD manager (provides global ordering + unique table).
    val_applicable : int
        Leaf node ID for "action is applicable".
    val_not_applicable : int
        Leaf node ID for "action is not applicable".

    Returns
    -------
    int
        Root node ID of the precondition FODD.
    """

    def _compile(precond: Any, if_true: int, if_false: int) -> int:
        """Recursively compile a precondition sub-tree."""
        if precond is None:
            return if_true  # No precondition = always applicable

        if not isinstance(precond, list) or len(precond) == 0:
            return if_true

        head = precond[0]

        # ── AND: chain of sequential tests ──
        if head == "and":
            # Build right-to-left: the last conjunct gates if_true/if_false,
            # each preceding conjunct gates the subsequent sub-diagram.
            node = if_true
            for child in reversed(precond[1:]):
                node = _compile(child, node, if_false)
            return node

        # ── OR: parallel branches ──
        if head == "or":
            # Build right-to-left: each disjunct can independently succeed.
            node = if_false
            for child in reversed(precond[1:]):
                node = _compile(child, if_true, node)
            return node

        # ── NOT: swap true/false branches ──
        if head == "not":
            return _compile(precond[1], if_false, if_true)

        # ── Numeric comparison — treat as always true for abstract purposes ──
        if head in (">", "<", ">=", "<=", "="):
            return if_true

        # ── Atomic predicate ──
        atom = map_pyppddl_to_pyrmdp(tuple(precond))
        return manager.get_node(atom, if_true, if_false)

    return _compile(action.precondition, val_applicable, val_not_applicable)


# ════════════════════════════════════════════════════════════════════
#  Effect → Leaf Encoding
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AbstractEffect:
    """
    An abstract effect: a set of predicates added and deleted,
    plus probability and reward delta.
    """
    add_predicates: FrozenSet[str]   # frozenset of predicate names
    del_predicates: FrozenSet[str]
    prob: float
    reward_delta: float = 0.0

    def effect_signature(self) -> str:
        """Unique string key for this abstract effect."""
        add_sorted = sorted(self.add_predicates)
        del_sorted = sorted(self.del_predicates)
        return f"+{add_sorted}-{del_sorted}"


def _extract_abstract_effects(action: ActionSchema) -> List[AbstractEffect]:
    """
    Convert an action's Effect list into AbstractEffect objects.
    At the abstract (lifted) level, we track predicate names only.
    """
    abstract_effects = []
    for eff in action.effects:
        add_preds = frozenset(p[0] for p in eff.add_predicates)
        del_preds = frozenset(p[0] for p in eff.del_predicates)
        
        reward_delta = 0.0
        for op, func_name, value in eff.numeric_effects:
            if func_name in ("reward", "total-reward"):
                if op == "increase":
                    reward_delta += value
                elif op == "decrease":
                    reward_delta -= value
        
        abstract_effects.append(AbstractEffect(
            add_predicates=add_preds,
            del_predicates=del_preds,
            prob=eff.prob,
            reward_delta=reward_delta,
        ))
    return abstract_effects


def build_effect_fodd(
    action: ActionSchema,
    manager: FODDManager,
) -> Tuple[int, Dict[int, AbstractEffect]]:
    """
    Build a simple FODD subtree encoding the probabilistic effects.

    For a deterministic action (1 effect), returns a single leaf.
    For a probabilistic action (N effects), encodes each outcome
    as a distinct leaf value.

    Returns
    -------
    (root_node_id, leaf_map) :
        root_node_id: The root of the effect sub-FODD.
        leaf_map: Maps leaf node ID → AbstractEffect for decoding.
    """
    abstract_effs = _extract_abstract_effects(action)
    leaf_map: Dict[int, AbstractEffect] = {}

    if len(abstract_effs) == 0:
        # No effects: leaf = 0 (no-op)
        leaf_id = manager.get_leaf(0.0)
        leaf_map[leaf_id] = AbstractEffect(
            add_predicates=frozenset(),
            del_predicates=frozenset(),
            prob=1.0,
        )
        return leaf_id, leaf_map

    if len(abstract_effs) == 1:
        # Deterministic: single leaf
        eff = abstract_effs[0]
        # Use a unique value based on effect signature
        val = hash(eff.effect_signature()) % 10000 + 1.0
        leaf_id = manager.get_leaf(val)
        leaf_map[leaf_id] = eff
        return leaf_id, leaf_map

    # Probabilistic: create distinct leaf for each outcome
    # We use the effect index as a unique value
    first_leaf = None
    for i, eff in enumerate(abstract_effs):
        val = float(i + 1)
        leaf_id = manager.get_leaf(val)
        leaf_map[leaf_id] = eff
        if first_leaf is None:
            first_leaf = leaf_id

    # For the lifted abstract representation, we represent all outcomes
    # under a single "applicable" subtree. The leaf_map tells us
    # which outcomes exist and their probabilities.
    # Return the first leaf as the representative; the full distribution
    # is captured by leaf_map.
    return first_leaf, leaf_map


# ════════════════════════════════════════════════════════════════════
#  Full Transition FODD: compose all actions
# ════════════════════════════════════════════════════════════════════

@dataclass
class ActionFODD:
    """A single action's FODD representation."""
    action_name: str
    root_id: int
    precondition_root: int
    leaf_map: Dict[int, AbstractEffect]
    abstract_effects: List[AbstractEffect]


def build_transition_fodd(
    actions: List[ActionSchema],
    manager: FODDManager,
) -> Tuple[int, List[ActionFODD]]:
    """
    Build a composed transition FODD for all actions in the domain.

    Each action is compiled into a precondition FODD with effect leaves.
    Actions are then composed via the `apply(max, ...)` operator —
    effectively creating a nondeterministic choice over all actions.

    Parameters
    ----------
    actions : List[ActionSchema]
        All actions from the pyPPDDL Domain.
    manager : FODDManager
        The FODD manager.

    Returns
    -------
    (composed_root, action_fodds) :
        composed_root: Root node ID of the composed transition FODD.
        action_fodds: Per-action FODD representations with leaf maps.
    """
    val_not_applicable = manager.get_leaf(0.0)
    action_fodds: List[ActionFODD] = []
    reducer = SyntacticReducer(manager)

    for action in actions:
        # Build effect sub-diagram
        effect_root, leaf_map = build_effect_fodd(action, manager)
        abstract_effs = _extract_abstract_effects(action)

        # Build precondition FODD: precond-True → effect_root, False → 0
        precond_root = build_precondition_fodd(
            action, manager, effect_root, val_not_applicable
        )

        action_fodds.append(ActionFODD(
            action_name=action.name,
            root_id=precond_root,
            precondition_root=precond_root,
            leaf_map=leaf_map,
            abstract_effects=abstract_effs,
        ))
        logger.info(
            f"Built FODD for action '{action.name}': "
            f"root={precond_root}, effects={len(abstract_effs)}"
        )

    # Compose all action FODDs via max (nondeterministic choice)
    if not action_fodds:
        return val_not_applicable, action_fodds

    composed = action_fodds[0].root_id
    for afodd in action_fodds[1:]:
        composed = apply(max, manager, composed, afodd.root_id)

    # Reduce the composed FODD
    [composed] = reducer.simplify([composed])
    logger.info(f"Composed transition FODD root: {composed}")

    return composed, action_fodds


# ════════════════════════════════════════════════════════════════════
#  Abstract State Enumeration from FODD Paths
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AbstractState:
    """
    An abstract state defined by a conjunction of predicate truth assignments.
    Represents one partition of the relational state space.
    """
    true_predicates: FrozenSet[str]   # predicate names known to be true
    false_predicates: FrozenSet[str]  # predicate names known to be false

    @property
    def label(self) -> str:
        parts = []
        for p in sorted(self.true_predicates):
            parts.append(p)
        for p in sorted(self.false_predicates):
            parts.append(f"¬{p}")
        return " ∧ ".join(parts) if parts else "∅"

    @property
    def short_id(self) -> str:
        """Short hash for graph node IDs."""
        h = hashlib.md5(self.label.encode()).hexdigest()[:6]
        return f"S_{h}"


def _enumerate_fodd_paths(
    node_id: int,
    manager: FODDManager,
    current_true: Set[str],
    current_false: Set[str],
) -> List[Tuple[FrozenSet[str], FrozenSet[str], float]]:
    """
    Recursively enumerate all root-to-leaf paths in a FODD.

    Returns a list of (true_preds, false_preds, leaf_value) tuples,
    one per path.
    """
    node = manager.nodes[node_id]

    if node.is_leaf:
        return [(frozenset(current_true), frozenset(current_false), node.value)]

    paths = []
    pred_name = node.query.predicate

    # High branch: predicate is true
    if node.high is not None:
        true_set = current_true | {pred_name}
        paths.extend(_enumerate_fodd_paths(
            node.high, manager, true_set, current_false
        ))

    # Low branch: predicate is false
    if node.low is not None:
        false_set = current_false | {pred_name}
        paths.extend(_enumerate_fodd_paths(
            node.low, manager, current_true, false_set
        ))

    return paths


def enumerate_abstract_states(
    action_fodds: List[ActionFODD],
    manager: FODDManager,
    domain: Domain,
) -> nx.DiGraph:
    """
    Enumerate all abstract states from the per-action FODDs and build
    the abstract transition graph (quotient graph).

    Each node is an AbstractState (conjunction of predicate truth values).
    Each edge represents a possible transition via some action, weighted
    by probability.

    Parameters
    ----------
    action_fodds : List[ActionFODD]
        Per-action FODD representations from build_transition_fodd().
    manager : FODDManager
        The FODD manager.
    domain : Domain
        The pyPPDDL Domain (for predicate names).

    Returns
    -------
    nx.DiGraph
        The abstract transition graph. Nodes have attributes:
          - 'state': AbstractState
          - 'label': human-readable label
        Edges have attributes:
          - 'action': action name
          - 'prob': transition probability
          - 'effect': AbstractEffect
    """
    G = nx.DiGraph()
    all_pred_names = {p.name for p in domain.predicates}

    for afodd in action_fodds:
        # Enumerate paths through this action's FODD
        paths = _enumerate_fodd_paths(
            afodd.precondition_root, manager, set(), set()
        )

        for true_preds, false_preds, leaf_value in paths:
            # Skip "not applicable" paths (leaf = 0)
            if leaf_value == 0.0:
                continue

            # The source abstract state
            source_state = AbstractState(
                true_predicates=true_preds,
                false_predicates=false_preds,
            )

            # Add source node
            src_id = source_state.short_id
            if src_id not in G:
                G.add_node(src_id, state=source_state, label=source_state.label)

            # Compute successor states for each effect
            for eff in afodd.abstract_effects:
                # Apply effect: add new predicates, remove deleted ones
                new_true = (true_preds | eff.add_predicates) - eff.del_predicates
                new_false = (false_preds | eff.del_predicates) - eff.add_predicates

                target_state = AbstractState(
                    true_predicates=new_true,
                    false_predicates=new_false,
                )

                tgt_id = target_state.short_id
                if tgt_id not in G:
                    G.add_node(tgt_id, state=target_state, label=target_state.label)

                # Add or update edge
                if G.has_edge(src_id, tgt_id):
                    # Accumulate probability
                    existing = G[src_id][tgt_id]
                    existing["prob"] = existing.get("prob", 0.0) + eff.prob
                    actions_set = existing.get("actions", set())
                    actions_set.add(afodd.action_name)
                    existing["actions"] = actions_set
                else:
                    G.add_edge(
                        src_id, tgt_id,
                        action=afodd.action_name,
                        actions={afodd.action_name},
                        prob=eff.prob,
                        effect=eff,
                    )

    logger.info(
        f"Abstract transition graph: {G.number_of_nodes()} states, "
        f"{G.number_of_edges()} transitions"
    )
    return G
