"""
Prompt Template — Step 0b: Task Description → PDDL Operators

Given:
  • The types and predicates produced by the VLM in Step 0a
  • One or more natural-language sentences describing the robot's task / policy

Ask an LLM to generate PDDL operators (actions) that are **directly specified
by the task sentences**.  Only operators traceable to an explicit phrase or
step in the task descriptions are generated — no extra "generally useful"
primitives.

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

Your job is to produce `:action` blocks for the operators that are
**directly described** by the task sentences.

Rules
─────
1. Each action must have `:parameters`, `:precondition`, and `:effect`.
2. Use ONLY the types and predicates already defined — do NOT invent new ones.
3. Effects are deterministic (no probabilities, no reward annotations).
4. Keep operators minimal and STRIPS-compatible (:strips :typing).
5. Use `(and …)` for conjunctions; `(not …)` for negation in effects.
6. Always reason about the `robot` type — actions should reference the robot
   performing them (e.g. `?r - robot` in parameters).
7. Output ONLY the `:action` blocks, one after another — nothing else.

CRITICAL CONSTRAINT:
- Generate ONLY operators that correspond to skills **explicitly mentioned
  or directly implied** by the task sentences.
- Do NOT generate extra "generally useful" operators that go beyond what
  the tasks describe. Each generated action must be traceable to a specific
  phrase or step in the task descriptions.
- For example, if the tasks say "pick up X and put it in Y", generate a
  pick-up action and a put-in action — but do NOT also generate place-on,
  stack, unstack, or other actions not mentioned in the tasks.
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Builder
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
Here is the partial PDDL domain (types and predicates only):

{domain_fragment}

The robot's task is described as follows:

{task_descriptions}

Generate the PDDL `:action` blocks for ONLY the operators that are directly
described or directly implied by the task sentences above.
Use ONLY the types and predicates above — do NOT add new ones.
Do NOT generate extra operators beyond what the tasks specify — each action
must be traceable to a specific phrase or step in the task description.

Respond with ONLY the action blocks (no explanations, no markdown), for example:

(:action pick-up
  :parameters (?r - robot ?x - movable ?s - surface)
  :precondition (and
    (clear ?x)
    (arm-empty ?r)
    (on ?x ?s)
  )
  :effect (and
    (holding ?r ?x)
    (not (clear ?x))
    (not (arm-empty ?r))
    (not (on ?x ?s))
  )
)

(:action place-on
  :parameters (?r - robot ?x - movable ?s - surface)
  :precondition (and
    (holding ?r ?x)
    (clear ?s)
  )
  :effect (and
    (on ?x ?s)
    (clear ?x)
    (arm-empty ?r)
    (not (holding ?r ?x))
    (not (clear ?s))
  )
)

(:action put-in
  :parameters (?r - robot ?m - movable ?c - container)
  :precondition (and
    (holding ?r ?m)
    (opened ?c)
  )
  :effect (and
    (in ?m ?c)
    (arm-empty ?r)
    (not (holding ?r ?m))
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

def _extract_all_balanced_sexps(text: str, opener: str) -> list[str]:
    """
    Extract every balanced S-expression whose opening tag matches *opener*.

    Uses parenthesis-depth tracking.  Quoted strings are treated as
    opaque so parentheses inside them are not counted.
    """
    results: list[str] = []
    search_from = 0
    while True:
        start = text.find(opener, search_from)
        if start < 0:
            break
        depth = 0
        in_string = False
        prev_ch = ""
        for i in range(start, len(text)):
            ch = text[i]
            if ch == '"' and prev_ch != "\\":
                in_string = not in_string
            elif not in_string:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        results.append(text[start : i + 1])
                        search_from = i + 1
                        break
            prev_ch = ch
        else:
            # Unbalanced — skip the truncated block rather than
            # including malformed PDDL that would break parsing.
            break
    return results


def parse_llm_operator_response(response_text: str) -> str:
    """
    Extract the PDDL action blocks from the LLM response.

    Handles models that emit "thinking" / chain-of-thought text mixed
    in with the PDDL.  Uses balanced-parenthesis extraction to pull out
    every ``(:action …)`` block.  Full domain assembly is handled by
    ``domain_genesis.py``.
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

    # If the response wraps everything in a (define …), work with the
    # inner content.
    if text.startswith("(define"):
        pass  # actions are inside; extraction below handles it

    # Extract all (:action …) blocks with balanced parens
    actions = _extract_all_balanced_sexps(text, "(:action")
    if actions:
        return "\n\n".join(actions)

    # Fallback: return cleaned text as-is
    return text
