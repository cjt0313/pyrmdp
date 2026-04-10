"""
Prompt Template — R5: LLM Mutex Constraint Generation

Given a list of PDDL predicate names from the domain, ask an LLM to
produce **mutex constraints** — pairs of predicates that cannot hold
simultaneously due to physical laws or logical necessity.

Three kinds of pairwise constraints are supported:

  • ``positive_mutex``:  pred_a and pred_b cannot both be TRUE.
  • ``negative_mutex``:  pred_a and pred_b cannot both be FALSE.
  • ``implication``:     If pred_a is TRUE then pred_b must be TRUE.

Additionally, **Exactly-One Mutex Groups** (SAS+ style) can be
identified:  a set of predicates applied to the same object arguments
where ONE AND ONLY ONE predicate is true at any time.  These groups
automatically imply both positive_mutex (no two can both be true) and
negative_mutex (no two can both be false — i.e. at least one holds)
for every pair in the group.

Used by :func:`pyrmdp.pruning.llm_axiom.generate_mutex_rules` and
:func:`pyrmdp.pruning.llm_axiom.generate_mutex_groups`.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Set

from .response_parser import extract_json_from_response

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  System Prompt  (pairwise constraints — legacy R5)
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a domain expert in robotic manipulation planning (PDDL).
Your job is to identify **mutex constraints** — pairs of predicates that
cannot hold simultaneously due to physical laws or logical necessity.

Three kinds of constraints are supported:
1. **positive_mutex**: pred_a and pred_b cannot BOTH be TRUE at the same
   time.
2. **negative_mutex**: pred_a and pred_b cannot BOTH be FALSE at the same
   time (at least one must hold).  Use sparingly.
3. **implication**: If pred_a is TRUE then pred_b MUST be TRUE.
   (Violation = pred_a true AND pred_b false.)

Be thorough but conservative — only include constraints that are physically
or logically NECESSARY.
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Template  (pairwise — legacy R5)
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
Given the following PDDL predicates from a robotic manipulation domain:

{predicate_list}

Generate a comprehensive list of mutex constraints for these predicates.

Produce ONLY valid JSON (no markdown, no commentary) with this exact schema:

{{
  "positive_mutex": [
    {{
      "pred_a": "holding",
      "pred_b": "arm-empty",
      "explanation": "The robot arm cannot hold something and be empty simultaneously."
    }}
  ],
  "negative_mutex": [
    {{
      "pred_a": "opened",
      "pred_b": "closed",
      "explanation": "A container must be either opened or closed."
    }}
  ],
  "implication": [
    {{
      "pred_a": "holding",
      "pred_b": "graspable",
      "explanation": "If holding an object, it must have been graspable."
    }}
  ]
}}

Include at least 5 positive_mutex rules and at least 3 implications.
Do NOT include trivially obvious predicates or numeric/reward predicates.
"""


# ════════════════════════════════════════════════════════════════════
#  System Prompt  (Exactly-One Mutex Groups — SAS+)
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_MUTEX_GROUPS = """\
You are a formal logic analyzer for PDDL domains.  Your task is to
analyze a list of PDDL predicates (with their typed parameters) and
group them into **Exactly-One Mutex Groups**.

An Exactly-One Mutex Group is a set of predicates applied to the same
(or overlapping) object arguments where **ONE AND ONLY ONE** predicate
can be true at any given time.  They are mutually exclusive *and*
exhaustive states of a single underlying property.

Examples of valid groups:
  • ["opened ?c", "closed ?c"]        — a container is either open or closed.
  • ["arm-empty ?a", "holding ?a ?o"] — the arm is either empty or holding something.

Guidelines:
  • Predicates in a group may have DIFFERENT arities as long as they
    share at least one object argument whose identity determines which
    predicate holds.  (e.g. arm-empty(?a) vs holding(?a, ?o) share ?a.)
  • Include the full predicate signature with typed variables.
  • Only include groups that are physically or logically NECESSARY.
  • Do NOT group predicates that are merely *correlated* but not truly
    mutually exclusive and exhaustive.
  • Ignore numeric/reward predicates.
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Template  (Exactly-One Mutex Groups)
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE_MUTEX_GROUPS = """\
Given the following PDDL predicates from a robotic manipulation domain:

{predicate_list}

Identify all Exactly-One Mutex Groups among these predicates.

Produce ONLY valid JSON (no markdown, no commentary) with this exact schema:

{{
  "mutex_groups": [
    {{
      "predicates": ["opened ?c", "closed ?c"],
      "shared_variable": "?c",
      "explanation": "A container is either open or closed — exactly one holds."
    }},
    {{
      "predicates": ["arm-empty ?a", "holding ?a ?o"],
      "shared_variable": "?a",
      "explanation": "A robot arm is either empty or holding an object — never both."
    }}
  ]
}}

Rules:
  • Each entry in "predicates" is the predicate name followed by its
    typed variable names, space-separated (e.g. "on-surface ?m ?s").
  • "shared_variable" is the variable that must be bound identically
    across all predicates in the group.
  • Only include groups that are truly mutually exclusive AND exhaustive.
  • It is fine if a predicate appears in zero groups.
"""


# ════════════════════════════════════════════════════════════════════
#  Prompt builders
# ════════════════════════════════════════════════════════════════════

def build_mutex_prompt(predicate_names: List[str]) -> Dict[str, str]:
    """
    Build the prompt payload for **pairwise** mutex constraint generation.

    Parameters
    ----------
    predicate_names : list[str]
        Predicate names from the PDDL domain.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready for the LLM.
    """
    pred_list_str = "\n".join(f"  - {p}" for p in sorted(predicate_names))
    user_text = USER_PROMPT_TEMPLATE.format(predicate_list=pred_list_str)
    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


def build_mutex_group_prompt(
    predicate_signatures: List[str],
) -> Dict[str, str]:
    """
    Build the prompt payload for **Exactly-One Mutex Group** detection.

    Parameters
    ----------
    predicate_signatures : list[str]
        Full predicate signatures including parameter names.
        Each entry should look like ``"opened ?c - container"`` or
        at minimum ``"opened ?c"``.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready for the LLM.
    """
    pred_list_str = "\n".join(
        f"  - {sig}" for sig in sorted(predicate_signatures)
    )
    user_text = USER_PROMPT_TEMPLATE_MUTEX_GROUPS.format(
        predicate_list=pred_list_str,
    )
    return {
        "system": SYSTEM_PROMPT_MUTEX_GROUPS,
        "user": user_text,
    }


# ════════════════════════════════════════════════════════════════════
#  Response parsers
# ════════════════════════════════════════════════════════════════════

def parse_mutex_response(
    response_text: str,
    valid_predicates: Set[str],
) -> Optional[List[Dict[str, str]]]:
    """
    Parse the LLM's JSON response for **pairwise** mutex constraints.

    Silently drops rules that reference unknown predicates or have
    ``pred_a == pred_b``.

    Parameters
    ----------
    response_text : str
        Raw LLM output.
    valid_predicates : set[str]
        Known predicate names — rules referencing unknown predicates
        are discarded.

    Returns
    -------
    list[dict] | None
        List of ``{"kind", "pred_a", "pred_b", "explanation"}`` dicts,
        or *None* on total parse failure.
    """
    data = extract_json_from_response(response_text)
    if data is None:
        logger.warning("No JSON found in LLM mutex response")
        return None

    rules: List[Dict[str, str]] = []
    for kind in ("positive_mutex", "negative_mutex", "implication"):
        for item in data.get(kind, []):
            pa = item.get("pred_a", "").strip()
            pb = item.get("pred_b", "").strip()
            expl = item.get("explanation", "")

            # Validate predicate names exist in the domain
            if pa not in valid_predicates:
                logger.debug("Skipping mutex: unknown predicate '%s'", pa)
                continue
            if pb not in valid_predicates:
                logger.debug("Skipping mutex: unknown predicate '%s'", pb)
                continue
            if pa == pb:
                continue  # self-mutex is meaningless

            rules.append({
                "kind": kind,
                "pred_a": pa,
                "pred_b": pb,
                "explanation": expl,
            })

    return rules


def parse_mutex_group_response(
    response_text: str,
    valid_predicates: Set[str],
) -> Optional[Dict[str, Any]]:
    """
    Parse the LLM's JSON response for **Exactly-One Mutex Groups**.

    Returns both the structured groups *and* pairwise rules (positive_mutex
    + negative_mutex for every pair in each group) so that the existing
    ``prune_with_mutexes()`` logic continues to work.

    Parameters
    ----------
    response_text : str
        Raw LLM output.
    valid_predicates : set[str]
        Known predicate names (bare names, without parameters) —
        groups referencing unknown predicates are discarded.

    Returns
    -------
    dict | None
        ``{"groups": list[dict], "pairwise_rules": list[dict]}``
        where each group is ``{"predicates": [...], "shared_variable": str,
        "explanation": str}`` and each pairwise rule is
        ``{"kind": str, "pred_a": str, "pred_b": str, "explanation": str}``.
        Returns *None* on total parse failure.
    """
    data = extract_json_from_response(response_text)
    if data is None:
        logger.warning("No JSON found in LLM mutex-group response")
        return None

    groups: List[Dict[str, Any]] = []
    pairwise_rules: List[Dict[str, str]] = []

    for item in data.get("mutex_groups", []):
        raw_preds: List[str] = item.get("predicates", [])
        shared_var: str = item.get("shared_variable", "")
        explanation: str = item.get("explanation", "")

        # Validate: extract bare predicate names and check them
        validated_sigs: List[str] = []
        for sig in raw_preds:
            sig = sig.strip()
            if not sig:
                continue
            bare_name = sig.split()[0]
            if bare_name not in valid_predicates:
                logger.debug(
                    "Dropping predicate '%s' from mutex group (not in domain)",
                    bare_name,
                )
                continue
            validated_sigs.append(sig)

        # Need at least 2 predicates for a meaningful group
        if len(validated_sigs) < 2:
            logger.debug(
                "Skipping mutex group with < 2 valid predicates: %s",
                raw_preds,
            )
            continue

        groups.append({
            "predicates": validated_sigs,
            "shared_variable": shared_var,
            "explanation": explanation,
        })

        # ── Emit pairwise rules for every combination ──
        # An Exactly-One group implies:
        #   • positive_mutex for every pair (no two can both be true)
        #   • negative_mutex for every pair (no two can both be false
        #     → at least one must hold)
        bare_names = [s.split()[0] for s in validated_sigs]
        for pa, pb in combinations(bare_names, 2):
            pairwise_rules.append({
                "kind": "positive_mutex",
                "pred_a": pa,
                "pred_b": pb,
                "explanation": f"Exactly-one group: {explanation}",
            })
            pairwise_rules.append({
                "kind": "negative_mutex",
                "pred_a": pa,
                "pred_b": pb,
                "explanation": f"Exactly-one group: {explanation}",
            })

    logger.info(
        "Parsed %d mutex groups → %d pairwise rules",
        len(groups), len(pairwise_rules),
    )

    return {
        "groups": groups,
        "pairwise_rules": pairwise_rules,
    }
