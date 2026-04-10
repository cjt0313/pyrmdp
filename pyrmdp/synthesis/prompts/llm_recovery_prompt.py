"""
Prompt Template — Step 5: LLM Recovery Operator Synthesis

Given a (sink SCC, source SCC) pair with their predicate truth
assignments, ask an LLM to synthesize a PPDDL operator that transitions
the world from the sink state toward the source state, plus a plausible
failure mode for the new operator.

The prompt asks for a structured JSON response with:
  • ``name`` — action name
  • ``parameters`` — typed PDDL parameters
  • ``preconditions`` — predicates that must hold
  • ``nominal_add/del`` — success-branch effects
  • ``failure_add/del`` — failure-branch effects
  • ``numeric_effects`` — reward adjustments
  • ``explanation`` — human-readable rationale

Used by :func:`pyrmdp.synthesis.delta_minimizer.delta_minimize`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from .response_parser import extract_json_from_response

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  System Prompt
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a PDDL domain expert specialising in recovery-action synthesis.
Given two abstract states from a condensed Markov chain (a *sink* state
and a *source* state), you design a PPDDL operator that transitions the
world from the sink toward the source.

Rules
─────
1. The operator must have typed parameters, preconditions satisfiable in
   the sink state, and nominal effects that produce (or move toward) the
   source state.
2. Hallucinate ONE physically plausible failure mode for the operator.
3. Keep the operator generalizable — use parametrised variables.
4. Respond with ONLY valid JSON (no explanatory prose outside the JSON).
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Template
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
Synthesize a PPDDL operator that transitions the world from State U
(a sink state) to State V (a source state).

State U (Sink SCC {sink_scc}):
  True predicates:  {sink_true}
  False predicates: {sink_false}

State V (Source SCC {source_scc}):
  True predicates:  {source_true}
  False predicates: {source_false}

Required change (delta): {delta} predicates must change.

Respond with ONLY valid JSON:
{{
    "name": "recover_action_name",
    "parameters": [{{"name": "?obj", "type": "physical-item"}}],
    "preconditions": [["pred_name", "?arg1", "?arg2"]],
    "nominal_add": [["pred_name", "?arg1"]],
    "nominal_del": [["pred_name", "?arg1"]],
    "failure_add": [["pred_name", "?arg1"]],
    "failure_del": [["pred_name", "?arg1"]],
    "numeric_effects": [["decrease", "reward", 1]],
    "explanation": "This recovery action does X."
}}
"""


# ════════════════════════════════════════════════════════════════════
#  Prompt builder
# ════════════════════════════════════════════════════════════════════

def build_recovery_prompt(
    *,
    sink_scc: int,
    source_scc: int,
    sink_true_preds: Set[str],
    sink_false_preds: Set[str],
    source_true_preds: Set[str],
    source_false_preds: Set[str],
    delta: int,
) -> Dict[str, str]:
    """
    Build the prompt payload for recovery-operator synthesis.

    Parameters
    ----------
    sink_scc / source_scc : int
        SCC node IDs in the condensation DAG.
    sink_true_preds / sink_false_preds : set[str]
        Predicate truth assignment in the sink SCC.
    source_true_preds / source_false_preds : set[str]
        Predicate truth assignment in the source SCC.
    delta : int
        Logical Hamming distance (predicate delta).

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready for the LLM.
    """
    user_text = USER_PROMPT_TEMPLATE.format(
        sink_scc=sink_scc,
        source_scc=source_scc,
        sink_true=sorted(sink_true_preds),
        sink_false=sorted(sink_false_preds),
        source_true=sorted(source_true_preds),
        source_false=sorted(source_false_preds),
        delta=delta,
    )
    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


# ════════════════════════════════════════════════════════════════════
#  Response parser
# ════════════════════════════════════════════════════════════════════

def parse_recovery_response(
    response_text: str,
    fallback_name: str = "recover",
) -> Optional[Dict[str, Any]]:
    """
    Parse the LLM's JSON response for a recovery-operator synthesis.

    Parameters
    ----------
    response_text : str
        Raw LLM output.
    fallback_name : str
        Default action name if the LLM omits one.

    Returns
    -------
    dict | None
        Parsed JSON with keys ``name``, ``parameters``,
        ``preconditions``, ``nominal_add``, ``nominal_del``,
        ``failure_add``, ``failure_del``, ``numeric_effects``,
        ``explanation``.  Returns *None* on parse failure.
    """
    data = extract_json_from_response(response_text)
    if data is None:
        logger.warning("No JSON found in LLM recovery-synthesis response")
        return None

    # Apply defaults for optional keys
    data.setdefault("name", fallback_name)
    data.setdefault("parameters", [])
    data.setdefault("preconditions", [])
    data.setdefault("nominal_add", [])
    data.setdefault("nominal_del", [])
    data.setdefault("failure_add", [])
    data.setdefault("failure_del", [])
    data.setdefault("numeric_effects", [])
    data.setdefault("explanation", "")

    return data
