"""
Prompt Template — Step 5b: LLM Feasibility & Ranking Gate

Given a Sink State and a batch of Candidate Source States, evaluates
whether each transition is physically achievable in one robotic skill
execution and ranks the feasible survivors.

Used by :func:`pyrmdp.synthesis.delta_minimizer.evaluate_candidates`.
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
You are a physics-aware evaluator for a robotic manipulation system.
You will be provided with a Sink State (the current physical situation) and a list of Candidate Source States (desired physical situations).

For each candidate transition (Sink -> Source), determine:
1. is_feasible (boolean): Is it physically possible for a standard robot arm to achieve this exact state transition directly in ONE skill execution, without violating physics (e.g., teleporting, phasing through solid matter)?
2. rank (integer 1 to k): For the candidates that ARE feasible, rank them from 1 (most natural, easiest robotic movement) to K (most awkward or complex). Use null if infeasible.

Output strictly in this JSON format:
{
  "evaluations": [
    {
      "candidate_id": 0,
      "is_feasible": true,
      "reasoning": "Moving from holding to placed is a standard action.",
      "rank": 1
    },
    {
      "candidate_id": 1,
      "is_feasible": false,
      "reasoning": "Cannot transition from dropped to inside-closed-box without the box being opened first.",
      "rank": null
    }
  ]
}
"""


# ════════════════════════════════════════════════════════════════════
#  User Prompt — Single-sink batch evaluation
# ════════════════════════════════════════════════════════════════════

_USER_SINGLE_SINK_TEMPLATE = """\
Evaluate the physical feasibility of the following candidate transitions.

## Sink State (Current Situation)
  True predicates:  {sink_true}
  False predicates: {sink_false}

## Candidate Source States
{candidates_block}

For each candidate, decide if a standard robot arm can achieve the \
transition from the Sink to that Source in ONE skill execution.
Rank all feasible candidates (1 = easiest, K = hardest). \
Use null for rank if infeasible.

Respond with ONLY valid JSON (no prose outside the JSON block).
"""

_CANDIDATE_ENTRY = """\
### Candidate {cid}
  True predicates:  {src_true}
  False predicates: {src_false}
  Predicate delta:  {delta} predicates must change
"""


# ════════════════════════════════════════════════════════════════════
#  User Prompt — Two-hop unroll verification
# ════════════════════════════════════════════════════════════════════

_USER_UNROLL_TEMPLATE = """\
Evaluate the physical feasibility of the following TWO-STEP recovery \
paths.  Each path routes through an intermediate state that already \
exists in the world model.

## Sink State (Current Situation)
  True predicates:  {sink_true}
  False predicates: {sink_false}

## Candidate Two-Step Paths
{paths_block}

For each candidate path, determine:
1. is_feasible (boolean): Can a robot execute BOTH steps sequentially \
(Sink -> Intermediate, then Intermediate -> Source) without violating physics?
2. rank (integer 1 to k): Rank the feasible paths (1 = easiest). Use null if infeasible.

Respond with ONLY valid JSON (no prose outside the JSON block):
{{
  "evaluations": [
    {{
      "candidate_id": 0,
      "is_feasible": true,
      "reasoning": "Both hops are standard manipulation actions.",
      "rank": 1
    }}
  ]
}}
"""

_UNROLL_PATH_ENTRY = """\
### Candidate {cid}
  Intermediate state:
    True predicates:  {mid_true}
    False predicates: {mid_false}
  Target source state:
    True predicates:  {src_true}
    False predicates: {src_false}
  Hop 1 delta (Sink → Intermediate): {delta1} predicates
  Hop 2 delta (Intermediate → Source): {delta2} predicates
"""


# ════════════════════════════════════════════════════════════════════
#  Prompt Builders
# ════════════════════════════════════════════════════════════════════

def build_feasibility_prompt(
    sink_true: Set[str],
    sink_false: Set[str],
    candidates: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Build a single-sink batch feasibility prompt.

    Parameters
    ----------
    sink_true, sink_false : set[str]
        Predicate assignments for the sink state.
    candidates : list[dict]
        Each dict has keys ``candidate_id``, ``source_true``,
        ``source_false``, ``delta``.

    Returns
    -------
    {"system": str, "user": str}
    """
    entries = []
    for c in candidates:
        entries.append(_CANDIDATE_ENTRY.format(
            cid=c["candidate_id"],
            src_true=sorted(c["source_true"]),
            src_false=sorted(c["source_false"]),
            delta=c["delta"],
        ))

    user = _USER_SINGLE_SINK_TEMPLATE.format(
        sink_true=sorted(sink_true),
        sink_false=sorted(sink_false),
        candidates_block="\n".join(entries),
    )
    return {"system": SYSTEM_PROMPT, "user": user}


def build_unroll_prompt(
    sink_true: Set[str],
    sink_false: Set[str],
    paths: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Build a two-hop unroll verification prompt.

    Parameters
    ----------
    sink_true, sink_false : set[str]
        Predicate assignments for the sink state.
    paths : list[dict]
        Each dict has keys ``candidate_id``, ``mid_true``, ``mid_false``,
        ``source_true``, ``source_false``, ``delta1``, ``delta2``.

    Returns
    -------
    {"system": str, "user": str}
    """
    entries = []
    for p in paths:
        entries.append(_UNROLL_PATH_ENTRY.format(
            cid=p["candidate_id"],
            mid_true=sorted(p["mid_true"]),
            mid_false=sorted(p["mid_false"]),
            src_true=sorted(p["source_true"]),
            src_false=sorted(p["source_false"]),
            delta1=p["delta1"],
            delta2=p["delta2"],
        ))

    user = _USER_UNROLL_TEMPLATE.format(
        sink_true=sorted(sink_true),
        sink_false=sorted(sink_false),
        paths_block="\n".join(entries),
    )
    return {"system": SYSTEM_PROMPT, "user": user}


# ════════════════════════════════════════════════════════════════════
#  Response Parser
# ════════════════════════════════════════════════════════════════════

def parse_feasibility_response(
    text: str,
    num_candidates: int,
) -> Optional[List[Dict[str, Any]]]:
    """Parse the LLM's JSON response into a list of evaluation dicts.

    Each returned dict has keys: ``candidate_id``, ``is_feasible``,
    ``reasoning``, ``rank``.  Missing or malformed entries are filled
    with ``is_feasible=False, rank=None``.

    Parameters
    ----------
    text : str
        Raw LLM response text.
    num_candidates : int
        Expected number of evaluations (for padding).

    Returns
    -------
    list[dict] | None
        Sorted by ``candidate_id``, or *None* if parsing fails entirely.
    """
    data = extract_json_from_response(text)
    if data is None:
        logger.warning("Feasibility response: no JSON found")
        return None

    evaluations = data.get("evaluations") if isinstance(data, dict) else None
    if not isinstance(evaluations, list):
        logger.warning("Feasibility response: missing 'evaluations' key")
        return None

    # Index by candidate_id for easy lookup
    by_id: Dict[int, Dict] = {}
    for ev in evaluations:
        if not isinstance(ev, dict):
            continue
        cid = ev.get("candidate_id")
        if cid is None:
            continue
        by_id[int(cid)] = {
            "candidate_id": int(cid),
            "is_feasible": bool(ev.get("is_feasible", False)),
            "reasoning": str(ev.get("reasoning", "")),
            "rank": ev.get("rank"),
        }

    # Build result list, padding missing entries as infeasible
    result = []
    for i in range(num_candidates):
        if i in by_id:
            result.append(by_id[i])
        else:
            result.append({
                "candidate_id": i,
                "is_feasible": False,
                "reasoning": "(missing from LLM response)",
                "rank": None,
            })

    return result
