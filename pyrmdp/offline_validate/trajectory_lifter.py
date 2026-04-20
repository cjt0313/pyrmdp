"""Existential abstraction: lift grounded empirical states to abstract signatures.

Strips object arguments from grounded predicates (e.g.,
``holding(robot, white_toy)`` → ``holding``) and applies existential
quantification (∃) to produce lifted boolean signatures that match the
pyrmdp FODD-based abstract graph format.

Existential rule:
    ∃x P(x) is True   ⟺  at least one grounding P(a) is True
    ¬∃x P(x) is True  ⟺  every grounding P(a) is False
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, FrozenSet, List, Tuple

LiftedState = Tuple[FrozenSet[str], FrozenSet[str]]


def strip_arguments(predicate: str) -> str:
    """``'holding(robot, cup)'`` → ``'holding'``."""
    idx = predicate.find("(")
    return predicate[:idx].strip() if idx >= 0 else predicate.strip()


def lift_state(
    grounded_state: Dict[str, bool],
) -> LiftedState:
    """Apply existential abstraction over grounded observations.

    Groups all grounded predicates by bare name, then:
      - ∃-True:  bare predicate has at least one True grounding
      - ∃-False: bare predicate has ALL groundings False

    A bare predicate can never appear in both sets.

    Returns
    -------
    tuple[frozenset[str], frozenset[str]]
        ``(existential_true, existential_false)`` of bare predicate names.
    """
    groups: Dict[str, List[bool]] = defaultdict(list)
    for pred, val in grounded_state.items():
        groups[strip_arguments(pred)].append(val)

    existential_true: set = set()
    existential_false: set = set()

    for bare, vals in groups.items():
        if any(vals):
            existential_true.add(bare)
        else:
            existential_false.add(bare)

    assert not (existential_true & existential_false), \
        f"Existential violation: {existential_true & existential_false}"

    return frozenset(existential_true), frozenset(existential_false)


def lift_trajectory(
    keyframes: List[Dict[str, bool]],
) -> List[LiftedState]:
    """Lift a sequence of grounded keyframes to abstract signatures."""
    return [lift_state(kf) for kf in keyframes]
