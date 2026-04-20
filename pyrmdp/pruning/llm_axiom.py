"""
R5 — LLM-based mutex axiom generation and abstract state pruning.

Instead of Z3, we query a language model with domain-expert knowledge to
produce **mutex constraints** (pairs of predicates that cannot be
simultaneously true/false).  These constraints then prune impossible
abstract states from the FODD-enumerated graph.

Three kinds of pairwise mutex rules are supported:

* **positive_mutex**: Two predicates cannot both be *true* in the same state.
  E.g. ``holding`` and ``arm-empty`` are mutually exclusive.
* **implication**: If predicate A is true, predicate B must also be true.
  E.g. ``holding → ¬arm-empty`` (captured as positive_mutex above),
  or ``on(x,y) → ¬clear(y)``.
* **negative_mutex**: Two predicates cannot both be *false* simultaneously
  (at least one must hold).  Rarer in practice.

Additionally, **Exactly-One Mutex Groups** (SAS+ style) provide richer
structure: a set of predicates where ONE AND ONLY ONE is true at any
time for the same bound object(s).  These are used for:

1. **Operator auto-patching** — if an action adds ``opened ?c``, we
   automatically inject ``(not (closed ?c))`` into delete effects.
2. **Distance correction** — switching between members of the same
   group costs δ=1 instead of the naïve δ=2.
3. **State pruning** — every pair gets both positive_mutex and
   negative_mutex rules.

Usage
-----
>>> from pyrmdp.pruning.llm_axiom import generate_mutex_rules, prune_with_mutexes
>>> rules = generate_mutex_rules(predicate_names, llm_fn=my_fn)
>>> pruned_graph = prune_with_mutexes(abstract_graph, rules)

>>> from pyrmdp.pruning.llm_axiom import generate_mutex_groups, patch_operator_effects
>>> groups, extra_rules = generate_mutex_groups(pred_sigs, llm_fn=my_fn)
>>> patched_action = patch_operator_effects(action_schema, groups, domain)
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import networkx as nx

from pyrmdp.synthesis.prompts.llm_mutex_prompt import (
    build_mutex_prompt,
    build_mutex_group_prompt,
    parse_mutex_response as _parse_mutex_prompt_response,
    parse_mutex_group_response as _parse_mutex_group_prompt_response,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Data model — pairwise constraints
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MutexRule:
    """A single mutex constraint between predicates.

    Attributes
    ----------
    kind : str
        One of ``"positive_mutex"``, ``"negative_mutex"``, or ``"implication"``.
    pred_a : str
        First predicate name.
    pred_b : str
        Second predicate name.
    explanation : str
        Human-readable reason (from the LLM).
    """
    kind: str          # "positive_mutex" | "negative_mutex" | "implication"
    pred_a: str
    pred_b: str
    explanation: str = ""

    def violates(
        self,
        true_preds: FrozenSet[str],
        false_preds: FrozenSet[str],
    ) -> bool:
        """Return True if the given truth assignment violates this rule."""
        if self.kind == "positive_mutex":
            # Cannot both be true
            return self.pred_a in true_preds and self.pred_b in true_preds

        if self.kind == "negative_mutex":
            # Cannot both be false (at least one must hold)
            return self.pred_a in false_preds and self.pred_b in false_preds

        if self.kind == "implication":
            # pred_a → pred_b : if A is true then B must be true
            # violation: A true AND B false
            return self.pred_a in true_preds and self.pred_b in false_preds

        return False


# ════════════════════════════════════════════════════════════════════
#  Data model — Exactly-One Mutex Groups  (SAS+)
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExactlyOneGroup:
    """An exactly-one mutex group (SAS+-style multi-valued variable).

    All member predicates share at least one object argument and
    exactly **one** must be true at all times.

    Attributes
    ----------
    predicates : tuple[str, ...]
        Bare predicate names in this group (e.g. ``("opened", "closed")``).
    signatures : tuple[str, ...]
        Full signatures as returned by the LLM (e.g.
        ``("opened ?c", "closed ?c")``).
    shared_variable : str
        The typed variable shared by all members (e.g. ``"?c"``).
    explanation : str
        Human-readable reason from the LLM.
    """
    predicates: Tuple[str, ...]
    signatures: Tuple[str, ...]
    shared_variable: str = ""
    explanation: str = ""

    # ── Predicate → variable mapping (cached) ──────────────────────

    def _sig_map(self) -> Dict[str, List[str]]:
        """Return ``{pred_name: [var1, var2, ...]}`` from signatures."""
        result: Dict[str, List[str]] = {}
        for sig in self.signatures:
            parts = sig.split()
            name = parts[0]
            args = [p for p in parts[1:] if p.startswith("?")]
            result[name] = args
        return result

    def violates(
        self,
        true_preds: FrozenSet[str],
        false_preds: FrozenSet[str],
    ) -> bool:
        """Return True if the truth assignment violates exactly-one.

        Violation means:
          • More than one member is true, OR
          • All members are false.
        """
        true_count = sum(1 for p in self.predicates if p in true_preds)
        if true_count > 1:
            return True
        # All false — but only if we can verify (every member known)
        all_known = all(
            p in true_preds or p in false_preds for p in self.predicates
        )
        if all_known and true_count == 0:
            return True
        return False


@dataclass
class MutexResult:
    """Result of mutex generation + pruning."""
    rules: List[MutexRule]
    pruned_count: int = 0
    original_count: int = 0
    pruned_states: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
#  Prompt & Parser (delegated to prompts/ module)
# ════════════════════════════════════════════════════════════════════

def _build_mutex_prompt_str(predicate_names: List[str]) -> str:
    """Build a single-string prompt by concatenating system + user.

    Delegates to :mod:`synthesis.prompts.llm_mutex_prompt`.
    """
    parts = build_mutex_prompt(predicate_names)
    return parts["system"] + "\n\n" + parts["user"]


def _parse_mutex_response(text: str, valid_preds: Set[str]) -> List[MutexRule]:
    """Parse the LLM's JSON response into MutexRule objects.

    Delegates JSON extraction to
    :func:`synthesis.prompts.llm_mutex_prompt.parse_mutex_response`,
    then converts dicts to :class:`MutexRule` dataclasses.
    """
    raw_rules = _parse_mutex_prompt_response(text, valid_preds)
    if raw_rules is None:
        return []

    return [
        MutexRule(
            kind=r["kind"],
            pred_a=r["pred_a"],
            pred_b=r["pred_b"],
            explanation=r.get("explanation", ""),
        )
        for r in raw_rules
    ]


def _parse_mutex_group_response(
    text: str, valid_preds: Set[str],
) -> Tuple[List[ExactlyOneGroup], List[MutexRule]]:
    """Parse the LLM's JSON response for mutex groups.

    Returns
    -------
    groups : list[ExactlyOneGroup]
    pairwise_rules : list[MutexRule]
        Both positive_mutex AND negative_mutex for every pair in each group.
    """
    parsed = _parse_mutex_group_prompt_response(text, valid_preds)
    if parsed is None:
        return [], []

    groups: List[ExactlyOneGroup] = []
    for g in parsed["groups"]:
        sigs = tuple(g["predicates"])
        bare_names = tuple(s.split()[0] for s in sigs)
        groups.append(ExactlyOneGroup(
            predicates=bare_names,
            signatures=sigs,
            shared_variable=g.get("shared_variable", ""),
            explanation=g.get("explanation", ""),
        ))

    pairwise_rules: List[MutexRule] = [
        MutexRule(
            kind=r["kind"],
            pred_a=r["pred_a"],
            pred_b=r["pred_b"],
            explanation=r.get("explanation", ""),
        )
        for r in parsed["pairwise_rules"]
    ]

    return groups, pairwise_rules


# ════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════

def generate_mutex_rules(
    predicate_names: List[str],
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
    vlm_fn: Optional[Callable] = None,
    image_paths: Optional[List[Union[str, Path]]] = None,
) -> List[MutexRule]:
    """
    Query an LLM for domain mutex constraints on the given predicates.

    Parameters
    ----------
    predicate_names : list[str]
        Predicate names from the PDDL domain.
    llm_fn : callable, optional
        ``fn(prompt) → response_text``.  If *None*, builds one from
        ``llm.yaml`` via :func:`synthesis.llm_config.build_llm_fn`.
    vlm_fn : callable, optional
        ``fn(system, user_text, images) → response_text``.  Used when
        *image_paths* is provided.  If *None* and images are given,
        builds one from ``domain_genesis._build_default_vlm_fn``.
    image_paths : list[str | Path], optional
        Scene images to ground the mutex generation via VLM.

    Returns
    -------
    list[MutexRule]
        Validated mutex rules referencing only known predicates.
    """
    use_vlm = bool(image_paths)

    logger.info("Generating mutex rules for %d predicates …", len(predicate_names))

    if use_vlm:
        from pyrmdp.synthesis.domain_genesis import (
            _load_image_as_base64,
            _image_mime,
            _build_default_vlm_fn,
        )

        if vlm_fn is None:
            vlm_fn = _build_default_vlm_fn()

        parts = build_mutex_prompt(predicate_names, use_vlm=True)
        images = [
            {"mime": _image_mime(p), "base64": _load_image_as_base64(p)}
            for p in image_paths
        ]
        logger.info("  VLM mutex: sending %d image(s)", len(images))
        response = vlm_fn(parts["system"], parts["user"], images)
    else:
        if llm_fn is None:
            try:
                from pyrmdp.synthesis.llm_config import build_llm_fn
                llm_fn = build_llm_fn()
            except (EnvironmentError, ImportError) as exc:
                logger.warning("LLM unavailable (%s) — returning empty rules.", exc)
                return []

        prompt = _build_mutex_prompt_str(predicate_names)
        response = llm_fn(prompt)

    logger.debug("Mutex LLM response:\n%s", response)

    valid_preds = set(predicate_names)
    rules = _parse_mutex_response(response, valid_preds)

    logger.info(
        "Parsed %d mutex rules: %d positive_mutex, %d negative_mutex, %d implication",
        len(rules),
        sum(1 for r in rules if r.kind == "positive_mutex"),
        sum(1 for r in rules if r.kind == "negative_mutex"),
        sum(1 for r in rules if r.kind == "implication"),
    )
    return rules


def generate_mutex_groups(
    predicate_signatures: List[str],
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
    vlm_fn: Optional[Callable] = None,
    image_paths: Optional[List[Union[str, Path]]] = None,
    valid_predicate_names: Optional[Set[str]] = None,
) -> Tuple[List[ExactlyOneGroup], List[MutexRule]]:
    """
    Query an LLM for Exactly-One Mutex Groups.

    Parameters
    ----------
    predicate_signatures : list[str]
        Full predicate signatures, e.g. ``["opened ?c", "closed ?c",
        "holding ?a ?o", "arm-empty ?a"]``.
    llm_fn : callable, optional
        ``fn(prompt) → response_text``.
    vlm_fn : callable, optional
        ``fn(system, user_text, images) → response_text``.  Used when
        *image_paths* is provided.
    image_paths : list[str | Path], optional
        Scene images to ground the mutex generation via VLM.
    valid_predicate_names : set[str], optional
        Bare predicate names for validation.  If *None*, extracted from
        *predicate_signatures*.

    Returns
    -------
    groups : list[ExactlyOneGroup]
    pairwise_rules : list[MutexRule]
        Both ``positive_mutex`` and ``negative_mutex`` for every pair.
    """
    use_vlm = bool(image_paths)

    if valid_predicate_names is None:
        valid_predicate_names = {sig.split()[0] for sig in predicate_signatures}

    logger.info(
        "Generating mutex groups for %d predicates …",
        len(predicate_signatures),
    )

    if use_vlm:
        from pyrmdp.synthesis.domain_genesis import (
            _load_image_as_base64,
            _image_mime,
            _build_default_vlm_fn,
        )

        if vlm_fn is None:
            vlm_fn = _build_default_vlm_fn()

        parts = build_mutex_group_prompt(predicate_signatures, use_vlm=True)
        images = [
            {"mime": _image_mime(p), "base64": _load_image_as_base64(p)}
            for p in image_paths
        ]
        logger.info("  VLM mutex groups: sending %d image(s)", len(images))
        response = vlm_fn(parts["system"], parts["user"], images)
    else:
        if llm_fn is None:
            try:
                from pyrmdp.synthesis.llm_config import build_llm_fn
                llm_fn = build_llm_fn()
            except (EnvironmentError, ImportError) as exc:
                logger.warning("LLM unavailable (%s) — returning empty groups.", exc)
                return [], []

        parts = build_mutex_group_prompt(predicate_signatures)
        prompt = parts["system"] + "\n\n" + parts["user"]
        response = llm_fn(prompt)

    logger.debug("Mutex-group LLM response:\n%s", response)

    groups, pairwise_rules = _parse_mutex_group_response(
        response, valid_predicate_names,
    )

    logger.info(
        "Parsed %d mutex groups → %d pairwise rules",
        len(groups), len(pairwise_rules),
    )
    return groups, pairwise_rules


# ════════════════════════════════════════════════════════════════════
#  Operator auto-patching with mutex groups
# ════════════════════════════════════════════════════════════════════

def _bind_params(
    pred_sig_vars: List[str],
    action_params: List[Any],
    domain_predicates: Dict[str, Any],
    pred_name: str,
) -> Optional[List[str]]:
    """Try to bind a predicate's variables to the action's parameters.

    Works by matching **types**: for each variable in the predicate's
    signature we find the corresponding action parameter of the same
    PDDL type.

    Parameters
    ----------
    pred_sig_vars : list[str]
        Variable names from the group signature (e.g. ``["?c"]``).
    action_params : list[TypedParam]
        The action's ``parameters`` (from pyPPDDL ``ActionSchema``).
    domain_predicates : dict[str, Predicate]
        Predicate definitions keyed by name.
    pred_name : str
        The predicate we're trying to bind (used for type lookup).

    Returns
    -------
    list[str] | None
        Bound variable names from the action's parameter list,
        or *None* if binding fails.
    """
    pred_def = domain_predicates.get(pred_name)
    if pred_def is None:
        return None

    # Build {var_name: type} for the predicate definition
    # pred_def.parameters is list[TypedParam] with .name and .type attrs
    pred_param_types: Dict[str, str] = {}
    for pp in pred_def.parameters:
        pred_param_types[pp.name] = pp.type

    # Build {type: [action_var_name, ...]} from the action's params
    action_type_map: Dict[str, List[str]] = {}
    for ap in action_params:
        action_type_map.setdefault(ap.type, []).append(ap.name)

    bound: List[str] = []
    for var in pred_sig_vars:
        needed_type = pred_param_types.get(var)
        if needed_type is None:
            return None  # Can't resolve type
        candidates = action_type_map.get(needed_type, [])
        if not candidates:
            return None  # Action lacks this type
        # Take the first candidate (stable ordering)
        bound.append(candidates[0])

    return bound


def patch_operator_effects(
    action_schema: Any,
    groups: List[ExactlyOneGroup],
    domain: Any,
) -> Any:
    """
    Auto-patch an action's effects to respect exactly-one mutex groups.

    For each effect branch:
      • If predicate P is in ``add_predicates`` and P belongs to group G,
        inject ``(not Q bound_args)`` into ``del_predicates`` for every
        other member Q of G — **if** the action has matching typed
        parameters.
      • If binding fails (mixed-arity + action lacks needed params),
        log a warning and skip that group member.

    Parameters
    ----------
    action_schema : pyPPDDL ActionSchema
        The action to patch.  **NOT modified in-place** — a deep copy is
        returned.
    groups : list[ExactlyOneGroup]
        Exactly-one mutex groups.
    domain : pyPPDDL Domain
        The parsed domain, used for predicate type information.

    Returns
    -------
    ActionSchema
        A (possibly mutated) deep copy of the input.
    """
    if not groups:
        return action_schema

    # Build predicate lookup  {name: Predicate}
    domain_predicates: Dict[str, Any] = {
        p.name: p for p in domain.predicates
    }

    # Build group lookup  {pred_name: [group, ...]}
    pred_to_groups: Dict[str, List[ExactlyOneGroup]] = {}
    for g in groups:
        for pname in g.predicates:
            pred_to_groups.setdefault(pname, []).append(g)

    patched = copy.deepcopy(action_schema)
    action_params = patched.parameters

    for eff in patched.effects:
        # Collect all predicates being added in this effect branch
        existing_adds = {t[0] for t in eff.add_predicates}
        existing_dels = {t[0] for t in eff.del_predicates}
        new_dels: List[Tuple] = []

        for add_tuple in list(eff.add_predicates):
            added_pred = add_tuple[0]
            added_args = list(add_tuple[1:])

            for grp in pred_to_groups.get(added_pred, []):
                # For every OTHER member of this group, inject a delete
                for other_pred in grp.predicates:
                    if other_pred == added_pred:
                        continue
                    # Already explicitly deleted?
                    if other_pred in existing_dels:
                        continue

                    # Bind the other predicate's variables
                    other_sig = None
                    for sig in grp.signatures:
                        if sig.split()[0] == other_pred:
                            other_sig = sig
                            break

                    if other_sig is None:
                        continue

                    other_vars = [
                        p for p in other_sig.split()[1:] if p.startswith("?")
                    ]

                    bound = _bind_params(
                        other_vars, action_params,
                        domain_predicates, other_pred,
                    )

                    if bound is None:
                        logger.warning(
                            "patch_operator_effects: cannot bind '%s' in "
                            "action '%s' (missing typed param) — skipping",
                            other_pred, patched.name,
                        )
                        continue

                    # Build the delete tuple: (pred_name, ?v1, ?v2, ...)
                    del_tuple = tuple([other_pred] + bound)
                    if del_tuple not in {tuple(d) for d in eff.del_predicates}:
                        new_dels.append(del_tuple)
                        existing_dels.add(other_pred)

        if new_dels:
            eff.del_predicates = list(eff.del_predicates) + new_dels
            logger.debug(
                "Patched action '%s': injected %d mutex deletes: %s",
                patched.name, len(new_dels), new_dels,
            )

    return patched


def prune_with_mutexes(
    graph: nx.DiGraph,
    rules: List[MutexRule],
) -> MutexResult:
    """
    Remove abstract states from *graph* that violate any mutex rule.

    Operates **in-place** on the graph (removes nodes and their edges).

    Parameters
    ----------
    graph : nx.DiGraph
        The abstract transition graph from ``enumerate_abstract_states()``.
        Nodes must have a ``'state'`` attribute with ``.true_predicates``
        and ``.false_predicates`` frozensets.
    rules : list[MutexRule]
        Mutex rules from ``generate_mutex_rules()``.

    Returns
    -------
    MutexResult
        Summary of what was pruned.
    """
    if not rules:
        return MutexResult(
            rules=rules,
            pruned_count=0,
            original_count=graph.number_of_nodes(),
        )

    original_count = graph.number_of_nodes()
    to_remove: List[str] = []

    for nid, data in graph.nodes(data=True):
        state = data.get("state")
        if state is None:
            continue

        tp = state.true_predicates
        fp = state.false_predicates

        for rule in rules:
            if rule.violates(tp, fp):
                to_remove.append(nid)
                logger.debug(
                    "Pruning state %s: violates %s(%s, %s) — %s",
                    nid, rule.kind, rule.pred_a, rule.pred_b,
                    rule.explanation,
                )
                break  # One violation is enough

    for nid in to_remove:
        graph.remove_node(nid)

    pruned_count = len(to_remove)
    logger.info(
        "R5 mutex pruning: %d → %d states (-%d pruned)",
        original_count, graph.number_of_nodes(), pruned_count,
    )

    return MutexResult(
        rules=rules,
        pruned_count=pruned_count,
        original_count=original_count,
        pruned_states=to_remove,
    )


def rules_to_dict(rules: List[MutexRule]) -> List[Dict[str, str]]:
    """Serialize mutex rules to JSON-friendly dicts."""
    return [
        {
            "kind": r.kind,
            "pred_a": r.pred_a,
            "pred_b": r.pred_b,
            "explanation": r.explanation,
        }
        for r in rules
    ]


def rules_from_dict(data: List[Dict[str, str]]) -> List[MutexRule]:
    """Deserialize mutex rules from dicts."""
    return [
        MutexRule(
            kind=d["kind"],
            pred_a=d["pred_a"],
            pred_b=d["pred_b"],
            explanation=d.get("explanation", ""),
        )
        for d in data
    ]


def groups_to_dict(groups: List[ExactlyOneGroup]) -> List[Dict[str, Any]]:
    """Serialize ExactlyOneGroups to JSON-friendly dicts."""
    return [
        {
            "predicates": list(g.predicates),
            "signatures": list(g.signatures),
            "shared_variable": g.shared_variable,
            "explanation": g.explanation,
        }
        for g in groups
    ]


def groups_from_dict(data: List[Dict[str, Any]]) -> List[ExactlyOneGroup]:
    """Deserialize ExactlyOneGroups from dicts."""
    return [
        ExactlyOneGroup(
            predicates=tuple(d["predicates"]),
            signatures=tuple(d.get("signatures", d["predicates"])),
            shared_variable=d.get("shared_variable", ""),
            explanation=d.get("explanation", ""),
        )
        for d in data
    ]


# ════════════════════════════════════════════════════════════════════
#  Legacy API (kept for backward compat)
# ════════════════════════════════════════════════════════════════════

def generate_background_knowledge(
    predicates_list: list,
    *,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Legacy wrapper.  Prefer :func:`generate_mutex_rules` instead.
    """
    rules = generate_mutex_rules(predicates_list, llm_fn=llm_fn)
    lines = []
    for r in rules:
        if r.kind == "positive_mutex":
            lines.append(
                f"FORALL x: NOT ({r.pred_a}(x) AND {r.pred_b}(x))"
            )
        elif r.kind == "implication":
            lines.append(
                f"FORALL x: IF {r.pred_a}(x) THEN {r.pred_b}(x)"
            )
        elif r.kind == "negative_mutex":
            lines.append(
                f"FORALL x: {r.pred_a}(x) OR {r.pred_b}(x)"
            )
    return "\n".join(lines)
