"""
Prompt Template — Step 0b: Task Description → PDDL Operators

Given:
  • The types and predicates produced by the VLM in Step 0a
  • One or more natural-language sentences describing the robot's task / policy

Ask an LLM to generate PDDL operators (actions) with typed parameters,
preconditions, and effects.

The generated operators are purely deterministic at this stage — failure
branches are injected later (Step 1: ``llm_failure.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
#  System Prompt
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a PDDL domain engineer specialising in robotics manipulation.
You will be given:
  1. A partial PDDL domain containing ONLY (:types …) and (:predicates …).
  2. One or more natural-language descriptions of tasks the robot should perform.

Your job is to produce the complete set of `:action` blocks that would
enable a planner to achieve the described tasks.

Rules
─────
1. Each action must have `:parameters`, `:precondition`, and `:effect`.
2. Use ONLY the types and predicates already defined — do NOT invent new ones.
3. Effects are deterministic (no probabilities, no reward annotations).
4. Keep operators minimal and STRIPS-compatible (:strips :typing).
5. Use `(and …)` for conjunctions; `(not …)` for negation in effects.
6. Prefer one action per distinct physical skill (e.g. pick, place, push).
7. Always reason about the `robot` type — actions should reference the robot
   performing them (e.g. `?r - robot` in parameters).
8. Output ONLY the `:action` blocks, one after another — nothing else.
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Builder
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
Here is the partial PDDL domain (types and predicates only):

{domain_fragment}

The robot's task is described as follows:

{task_descriptions}

Generate the PDDL `:action` blocks needed to accomplish the described task(s).
Use ONLY the types and predicates above — do NOT add new ones.

Respond with ONLY the action blocks, for example:

(:action pick-up
  :parameters (?r - robot ?x - movable)
  :precondition (and
    (clear ?x)
    (arm-empty ?r)
    (on-table ?x)
  )
  :effect (and
    (holding ?r ?x)
    (not (clear ?x))
    (not (arm-empty ?r))
    (not (on-table ?x))
  )
)

(:action place-on
  :parameters (?r - robot ?x - movable ?y - object)
  :precondition (and
    (holding ?r ?x)
    (clear ?y)
  )
  :effect (and
    (on ?x ?y)
    (clear ?x)
    (arm-empty ?r)
    (not (holding ?r ?x))
    (not (clear ?y))
  )
)
"""


def build_llm_operator_prompt(
    domain_fragment: str,
    task_descriptions: List[str],
    extra_context: Optional[str] = None,
) -> dict:
    """
    Build the prompt payload for operator generation.

    Parameters
    ----------
    domain_fragment : str
        The PDDL ``(:types …)`` and ``(:predicates …)`` text produced by
        Step 0a (VLM).  Can also be a full ``(define (domain …) …)`` block
        — the LLM will use what it needs.
    task_descriptions : list[str]
        One or more natural-language sentences describing the tasks / policy.
        Example: ``["Stack all red blocks on the blue plate.",
                     "If a block falls, pick it up and retry."]``
    extra_context : str, optional
        Additional context (e.g. environment constraints) appended to the
        user message.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready to be sent to the LLM.
    """
    task_block = "\n".join(f"  - {t}" for t in task_descriptions)

    user_text = USER_PROMPT_TEMPLATE.format(
        domain_fragment=domain_fragment,
        task_descriptions=task_block,
    )

    if extra_context:
        user_text += f"\nAdditional context:\n{extra_context}\n"

    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


# ════════════════════════════════════════════════════════════════════
#  Response parser
# ════════════════════════════════════════════════════════════════════

def parse_llm_operator_response(response_text: str) -> str:
    """
    Extract the PDDL action blocks from the LLM response.

    Strips surrounding markdown fences and returns the raw PDDL string
    containing one or more ``(:action …)`` blocks.  Full domain assembly
    is handled by ``domain_genesis.py``.
    """
    text = response_text.strip()

    # Strip markdown code fences
    for lang_tag in ("```pddl", "```lisp", "```"):
        if text.startswith(lang_tag):
            text = text[len(lang_tag):]
            break
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # If the response wraps everything in a (define …), extract just the
    # action blocks.
    if text.startswith("(define"):
        import re
        actions = re.findall(r"\(:action\s[\s\S]*?\n\)", text)
        if actions:
            text = "\n\n".join(actions)

    return text
