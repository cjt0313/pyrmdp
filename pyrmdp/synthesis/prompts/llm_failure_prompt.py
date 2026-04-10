"""
Prompt Template — Step 1: LLM Failure Hallucination

For each operator in a PDDL domain, query an LLM to hallucinate a
physically plausible "worse effect" (failure mode).  The failure may
introduce novel predicates and/or types that are absent from the
original domain.

The prompt asks for a structured JSON response with:
  • ``failure_add`` / ``failure_del`` — predicate-level effects
  • ``failure_numeric`` — reward / numeric adjustments
  • ``new_predicates`` / ``new_types`` — extensions to the domain
  • ``explanation`` — human-readable rationale

Used by :func:`pyrmdp.synthesis.llm_failure.hallucinate_failures`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .response_parser import extract_json_from_response

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  System Prompt
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a robotics domain expert specialising in PDDL failure-mode analysis.
Given a PDDL operator (with its precondition, effects, and the domain's
types/predicates), you hallucinate ONE physically plausible "worse effect"
(failure mode) that could occur when the action is executed by a robot.

Rules
─────
1. The failure must be physically plausible for a robotics scenario.
2. If the failure introduces a new physical state not captured by existing
   predicates, propose NEW predicates (with typed parameters) and/or new
   types.
3. The failure effect should be expressible as PDDL add/del effects.
4. Keep it minimal — one failure mode, not multiple.
5. The failure must be different from "no change" (unchanged).
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Template
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
{action_description}

Hallucinate ONE physically plausible failure mode for the action above.

Respond with ONLY valid JSON in this exact format:
{{
    "failure_add": [["pred_name", "?param1", "?param2"]],
    "failure_del": [["pred_name", "?param1"]],
    "failure_numeric": [["decrease", "reward", 10]],
    "new_predicates": [
        {{"name": "dropped", "parameters": [{{"name": "?obj", "type": "physical-item"}}]}}
    ],
    "new_types": {{"damaged-item": "physical-item"}},
    "explanation": "The robot could drop the object while manipulating it."
}}

If no new predicates or types are needed, use empty lists/dicts for those fields.
"""


# ════════════════════════════════════════════════════════════════════
#  Prompt builder
# ════════════════════════════════════════════════════════════════════

def build_failure_prompt(action_description: str) -> Dict[str, str]:
    """
    Build the prompt payload for failure hallucination.

    Parameters
    ----------
    action_description : str
        Human-readable serialisation of the PDDL action, including
        domain predicates, types, parameters, precondition, and effects.
        Typically produced by ``_format_action_for_prompt()`` in
        ``llm_failure.py``.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready for the LLM.
    """
    user_text = USER_PROMPT_TEMPLATE.format(
        action_description=action_description,
    )
    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


# ════════════════════════════════════════════════════════════════════
#  Response parser
# ════════════════════════════════════════════════════════════════════

def parse_failure_response(
    response_text: str,
    action_name: str = "<unknown>",
) -> Optional[Dict[str, Any]]:
    """
    Parse the LLM's JSON response for a failure hallucination.

    Parameters
    ----------
    response_text : str
        Raw LLM output.
    action_name : str
        Name of the action (for logging only).

    Returns
    -------
    dict | None
        Parsed JSON with keys ``failure_add``, ``failure_del``,
        ``failure_numeric``, ``new_predicates``, ``new_types``,
        ``explanation``.  Returns *None* on parse failure.
    """
    data = extract_json_from_response(response_text)
    if data is None:
        logger.warning("No JSON found in LLM response for %s", action_name)
        return None

    # Minimal validation — ensure the mandatory keys are present
    for key in ("failure_add", "failure_del"):
        if key not in data:
            data[key] = []

    return data
