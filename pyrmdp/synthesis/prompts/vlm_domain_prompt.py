"""
Prompt Template — Step 0a: VLM Scene → Object Types & Predicates

Given one or more RGB images of a scene, ask a Vision-Language Model (VLM) to
produce the *static* part of a PDDL domain:

  • Object types (always including a ``robot`` type)
  • Typed predicates that describe spatial / physical relations

**No operators are generated at this stage.**

The prompt is designed for models that accept images + text
(GPT-4o, Gemini, Claude 3, etc.).
"""

from __future__ import annotations

from typing import List, Optional


# ════════════════════════════════════════════════════════════════════
#  System Prompt
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a PDDL domain engineer specialising in robotics manipulation.
Your task is to observe a scene from one or more RGB images and produce
the *types* and *predicates* section of a PDDL domain file.

Rules
─────
1. Always include a `robot` type (the manipulator visible or implied).
2. Derive object types from what you see (e.g. block, cup, table, tray).
3. Use a type hierarchy when natural (e.g. `block - movable`, `movable - object`).
4. Predicates must be typed.  Prefer spatial / physical relations:
     (on ?x - movable ?y - surface)
     (holding ?r - robot ?x - movable)
     (clear ?x - surface)
     (arm-empty ?r - robot)
5. Do NOT generate any `:action` or `:operator` blocks.
6. Output ONLY a valid PDDL `(define (domain …) …)` block — nothing else.
"""

# ════════════════════════════════════════════════════════════════════
#  User Prompt Builder
# ════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
Observe the following image(s) of a robotic manipulation scene.

{scene_description}

Generate a PDDL domain file containing ONLY:
  • (:requirements :strips :typing)
  • (:types …)          — include `robot` as a type
  • (:predicates …)     — typed predicates describing relations in the scene

Do NOT include any :action blocks.

Respond with ONLY the PDDL domain block, e.g.:

(define (domain <name>)
  (:requirements :strips :typing)
  (:types
    robot - agent
    block cup - movable
    movable table - object
  )
  (:predicates
    (on ?x - movable ?y - object)
    (holding ?r - robot ?x - movable)
    (clear ?x - object)
    (arm-empty ?r - robot)
    (on-table ?x - movable ?t - table)
  )
)
"""


def build_vlm_domain_prompt(
    scene_description: str = "",
    domain_name_hint: Optional[str] = None,
) -> dict:
    """
    Build the prompt payload for the VLM domain-extraction query.

    Parameters
    ----------
    scene_description : str
        Optional extra text describing the scene (e.g. "A Franka robot
        next to a table with coloured blocks").  Inserted into the user
        message alongside the images.
    domain_name_hint : str, optional
        Suggested domain name.  Appended to the user message so the VLM
        uses it in ``(define (domain <name>) …)``.

    Returns
    -------
    dict
        ``{"system": str, "user": str}`` ready to be formatted into API
        messages.  The caller is responsible for attaching the actual
        image(s) in the format required by the VLM provider.
    """
    desc = scene_description or "See the attached image(s)."
    user_text = USER_PROMPT_TEMPLATE.format(scene_description=desc)

    if domain_name_hint:
        user_text += f"\nUse domain name: `{domain_name_hint}`.\n"

    return {
        "system": SYSTEM_PROMPT,
        "user": user_text,
    }


# ════════════════════════════════════════════════════════════════════
#  Response parser
# ════════════════════════════════════════════════════════════════════

def parse_vlm_domain_response(response_text: str) -> str:
    """
    Extract the PDDL domain block from the VLM response.

    Strips surrounding markdown fences if present and returns the raw
    PDDL string.  Validation / conversion to pyPPDDL objects is left
    to the caller (``domain_genesis.py``).
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

    # Sanity: must start with (define
    if not text.startswith("(define"):
        # Try to find it inside the response
        idx = text.find("(define")
        if idx >= 0:
            text = text[idx:]
        # else: return as-is; let caller deal with parse errors

    return text
