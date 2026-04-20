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
#  System Prompt  (pairwise constraints — VLM variant)
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_VLM = """\
You are a physics-aware logical constraint generator for robotic
manipulation.  You are provided with an image of a robotic workspace
and a list of PDDL predicates.  Your task is to identify pairwise
**mutex constraints** — pairs of predicates that cannot hold
simultaneously due to physical impossibility.

Three kinds of constraints are supported:
1. **positive_mutex**: pred_a and pred_b cannot BOTH be TRUE at the same
   time.
2. **negative_mutex**: pred_a and pred_b cannot BOTH be FALSE at the same
   time (at least one must hold).  Use sparingly.
3. **implication**: If pred_a is TRUE then pred_b MUST be TRUE.
   (Violation = pred_a true AND pred_b false.)

CRITICAL: You must base these constraints ONLY on what is physically
impossible in the provided image.  Do not generate mutexes for states
that might briefly overlap during a continuous physical action (e.g., a
robot's gripper closing around an object might make it both 'graspable'
and 'holding' simultaneously).  If a state overlap is physically
possible in reality, do NOT group them in a mutex.

IMPORTANT: The provided image shows only one static moment in time
(usually the initial state).  You must mentally project how the robot
will move through continuous space to manipulate these objects.
Do NOT generate a mutex rule just because two states don't overlap in
this specific picture.  You must ask yourself: "During the continuous,
fluid motion of a robot executing a task (e.g., the milliseconds while
fingers are closing, or an object is being released), is it physically
possible for these two predicates to overlap?"  If the answer is YES,
they are NOT mutually exclusive.

Be thorough but conservative — only include constraints that are
physically IMPOSSIBLE in the real world.
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
#  User Prompt Template  (pairwise — VLM variant)
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE_VLM = """\
Look at the provided image(s) of the robotic workspace.

Given the following PDDL predicates from a robotic manipulation domain:

{predicate_list}

Generate a list of mutex constraints for these predicates based on what
is **physically impossible** in this workspace.

Remember: during continuous robot motion (grasping, placing, opening),
predicates can briefly overlap.  Only flag a mutex if the overlap is
physically impossible at ANY point during a manipulation sequence.

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
      "pred_a": "inside",
      "pred_b": "closed",
      "explanation": "If an object is secured inside, the container must be closed."
    }}
  ]
}}

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
#  System Prompt  (Exactly-One Mutex Groups — VLM variant)
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_MUTEX_GROUPS_VLM = """\
You are a physics-aware logical constraint generator for robotic
manipulation.  You are provided with an image of a robotic workspace
and a list of PDDL predicates.  Your task is to identify
**Exactly-One Mutex Groups** — sets of predicates where exactly one
MUST be true, and all others MUST be false, applied to the same objects.

An Exactly-One Mutex Group is a set of predicates applied to the same
(or overlapping) object arguments where **ONE AND ONLY ONE** predicate
can be true at any given time.  They are mutually exclusive *and*
exhaustive states of a single underlying property.

Examples of valid groups:
  • ["opened ?c", "closed ?c"]        — a container is either open or closed.
  • ["arm-empty ?a", "holding ?a ?o"] — the arm is either empty or holding something.

CRITICAL: You must base these constraints ONLY on what is physically
impossible in the provided image.  Do not generate mutexes for states
that might briefly overlap during a continuous physical action (e.g., a
robot's gripper closing around an object might make it both 'graspable'
and 'holding' simultaneously).  If a state overlap is physically
possible in reality, do NOT group them in a mutex.

IMPORTANT: The provided image shows only one static moment in time
(usually the initial state).  You must mentally project how the robot
will move through continuous space to manipulate these objects.
Do NOT generate a mutex rule just because two states don't overlap in
this specific picture.  You must ask yourself: "During the continuous,
fluid motion of a robot executing a task (e.g., the milliseconds while
fingers are closing, or an object is being released), is it physically
possible for these two predicates to overlap?"  If the answer is YES,
they are NOT mutually exclusive.

Guidelines:
  • Predicates in a group may have DIFFERENT arities as long as they
    share at least one object argument whose identity determines which
    predicate holds.
  • Include the full predicate signature with typed variables.
  • Only include groups that are physically IMPOSSIBLE to overlap.
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
#  User Prompt Template  (Exactly-One Mutex Groups — VLM variant)
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE_MUTEX_GROUPS_VLM = """\
Look at the provided image(s) of the robotic workspace.

Given the following PDDL predicates from a robotic manipulation domain:

{predicate_list}

Identify all Exactly-One Mutex Groups among these predicates based on
what is **physically impossible** in this workspace.

Remember: during continuous robot motion (grasping, placing, opening),
predicates can briefly overlap.  Only create a group if the overlap is
physically impossible at ANY point during a manipulation sequence.

Produce ONLY valid JSON (no markdown, no commentary) with this exact schema:

{{
  "mutex_groups": [
    {{
      "predicates": ["opened ?c", "closed ?c"],
      "shared_variable": "?c",
      "explanation": "A container is either open or closed — exactly one holds."
    }}
  ]
}}

Rules:
  • Each entry in "predicates" is the predicate name followed by its
    typed variable names, space-separated (e.g. "on-surface ?m ?s").
  • "shared_variable" is the variable that must be bound identically
    across all predicates in the group.
  • Only include groups that are truly mutually exclusive AND exhaustive
    AND physically impossible to overlap during continuous motion.
  • It is fine if a predicate appears in zero groups.
"""


# ════════════════════════════════════════════════════════════════════
#  Prompt builders
# ════════════════════════════════════════════════════════════════════

def build_mutex_prompt(
    predicate_names: List[str],
    *,
    use_vlm: bool = False,
) -> Dict[str, str]:
    """
    Build the prompt payload for **pairwise** mutex constraint generation.

    Parameters
    ----------
    predicate_names : list[str]
        Predicate names from the PDDL domain.
    use_vlm : bool
        If *True*, return the VLM-aware (image-grounded) prompt variant.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready for the LLM/VLM.
    """
    pred_list_str = "\n".join(f"  - {p}" for p in sorted(predicate_names))
    sys_prompt = SYSTEM_PROMPT_VLM if use_vlm else SYSTEM_PROMPT
    usr_template = USER_PROMPT_TEMPLATE_VLM if use_vlm else USER_PROMPT_TEMPLATE
    user_text = usr_template.format(predicate_list=pred_list_str)
    return {
        "system": sys_prompt,
        "user": user_text,
    }


def build_mutex_group_prompt(
    predicate_signatures: List[str],
    *,
    use_vlm: bool = False,
) -> Dict[str, str]:
    """
    Build the prompt payload for **Exactly-One Mutex Group** detection.

    Parameters
    ----------
    predicate_signatures : list[str]
        Full predicate signatures including parameter names.
        Each entry should look like ``"opened ?c - container"`` or
        at minimum ``"opened ?c"``.
    use_vlm : bool
        If *True*, return the VLM-aware (image-grounded) prompt variant.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready for the LLM/VLM.
    """
    pred_list_str = "\n".join(
        f"  - {sig}" for sig in sorted(predicate_signatures)
    )
    sys_prompt = SYSTEM_PROMPT_MUTEX_GROUPS_VLM if use_vlm else SYSTEM_PROMPT_MUTEX_GROUPS
    usr_template = (
        USER_PROMPT_TEMPLATE_MUTEX_GROUPS_VLM if use_vlm
        else USER_PROMPT_TEMPLATE_MUTEX_GROUPS
    )
    user_text = usr_template.format(predicate_list=pred_list_str)
    return {
        "system": sys_prompt,
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
