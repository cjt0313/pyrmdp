"""Mutex-bounded temporal smoothing for VLM state sequences.

Filters out continuous 'blur' frames that violate Exactly-One mutex
groups (e.g., both ``opened`` and ``closed`` true during in-transit
motion), then collapses consecutive identical states into discrete
logical keyframes.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Loading mutex data from pipeline outputs
# ─────────────────────────────────────────────────────────────

def load_mutex_rules(pipeline_dir: str | Path) -> List[Dict[str, str]]:
    """Load pairwise mutex rules from the latest iteration's JSON."""
    import json

    d = Path(pipeline_dir)
    candidates = sorted(d.glob("iter*_step2_mutex_rules.json"))
    if not candidates:
        logger.warning("No mutex rule files found in %s", d)
        return []

    path = candidates[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data.get("rules", [])
    logger.info("Loaded %d mutex rules from %s", len(rules), path.name)
    return rules


def extract_exactly_one_groups(rules: List[Dict[str, str]]) -> List[FrozenSet[str]]:
    """Reconstruct Exactly-One groups from pairwise rules.

    Rules generated from ``ExactlyOneGroup`` carry explanations like
    ``"Exactly-one group: ..."`` and come in both ``positive_mutex``
    and ``negative_mutex`` flavours for each pair.  We cluster
    predicates that appear together in such pairs.
    """
    adj: Dict[str, Set[str]] = defaultdict(set)

    for r in rules:
        expl = r.get("explanation", "")
        if "exactly-one" not in expl.lower() and "exactly one" not in expl.lower():
            continue
        a, b = r["pred_a"], r["pred_b"]
        adj[a].add(b)
        adj[b].add(a)

    groups: List[FrozenSet[str]] = []
    visited: Set[str] = set()

    for seed in sorted(adj):
        if seed in visited:
            continue
        component: Set[str] = set()
        queue = [seed]
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            for nb in adj[node]:
                if nb not in component:
                    queue.append(nb)
        visited |= component
        groups.append(frozenset(component))

    logger.info("Extracted %d Exactly-One groups: %s", len(groups), groups)
    return groups


def _strip_args(pred: str) -> str:
    """``'holding(robot, cup)'`` → ``'holding'``."""
    idx = pred.find("(")
    return pred[:idx].strip() if idx >= 0 else pred.strip()


# ─────────────────────────────────────────────────────────────
#  Mutex violation checking
# ─────────────────────────────────────────────────────────────

def violates_mutex(
    state: Dict[str, bool],
    groups: List[FrozenSet[str]],
    pairwise_rules: List[Dict[str, str]],
) -> bool:
    """Return True if the state violates any mutex constraint.

    Works with grounded predicate keys — strips arguments to get
    bare names for matching against group members and pairwise rules.
    """
    bare_vals: Dict[str, bool] = {}
    for pred, val in state.items():
        bare = _strip_args(pred)
        if bare not in bare_vals:
            bare_vals[bare] = val
        elif val:
            bare_vals[bare] = True

    for group in groups:
        true_count = sum(1 for p in group if bare_vals.get(p, False))
        false_count = sum(1 for p in group if p in bare_vals and not bare_vals[p])
        if true_count > 1:
            return True
        if false_count == len(group) and len(group) > 1:
            return True

    for r in pairwise_rules:
        a, b, kind = r["pred_a"], r["pred_b"], r["kind"]
        va, vb = bare_vals.get(a), bare_vals.get(b)
        if va is None or vb is None:
            continue
        if kind == "positive_mutex" and va and vb:
            return True
        if kind == "negative_mutex" and not va and not vb:
            return True
        if kind == "implication" and va and not vb:
            return True

    return False


# ─────────────────────────────────────────────────────────────
#  Filter + collapse
# ─────────────────────────────────────────────────────────────

def _state_signature(state: Dict[str, bool]) -> Tuple[Tuple[str, bool], ...]:
    return tuple(sorted(state.items()))


def filter_and_collapse(
    states: List[Dict[str, bool]],
    groups: List[FrozenSet[str]],
    pairwise_rules: List[Dict[str, str]],
) -> List[Dict[str, bool]]:
    """Mutex-bounded temporal smoothing.

    1. Drop frames violating any mutex constraint (in-transit noise).
    2. Collapse consecutive identical states into single keyframes.
    """
    valid = []
    dropped = 0
    for s in states:
        if violates_mutex(s, groups, pairwise_rules):
            dropped += 1
        else:
            valid.append(s)

    logger.info(
        "Mutex filter: %d/%d frames valid (%d dropped as in-transit)",
        len(valid), len(states), dropped,
    )

    if not valid:
        return []

    keyframes = [valid[0]]
    for s in valid[1:]:
        if _state_signature(s) != _state_signature(keyframes[-1]):
            keyframes.append(s)

    logger.info(
        "Collapsed %d valid frames → %d discrete keyframes",
        len(valid), len(keyframes),
    )
    return keyframes
