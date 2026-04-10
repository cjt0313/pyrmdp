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

import re
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
4. Predicates must be typed and MUST cover THREE perspectives:

   (a) **Object-self** — intrinsic / unary properties of a single object:
         (opened ?c - container)
         (closed ?c - container)
         (empty ?c - container)
         (broken ?m - movable)
         (upright ?m - movable)
         (graspable ?m - movable)

   (b) **Object-object** — binary relations between two objects:
         (on ?x - movable ?y - object)
         (inside ?x - movable ?c - container)
         (holding ?r - robot ?x - movable)
         (stacked ?x - movable ?y - movable)
         (adjacent ?x - object ?y - object)
         (heavier-than ?x - movable ?y - movable)

   (c) **Object-environment** — relations between objects and the workspace:
         (on-table ?x - movable ?t - surface)
         (clear ?x - object)
         (arm-empty ?r - robot)
         (at-location ?x - object ?l - location)
         (reachable ?r - robot ?x - object)
         (in-workspace ?m - movable)

   Generate predicates from ALL three categories.  Aim for at least
   2-3 predicates per category.

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
  • (:predicates …)     — typed predicates from THREE perspectives:
      (a) object-self: intrinsic / state properties (e.g. opened, closed, empty, upright)
      (b) object-object: relations between objects (e.g. on, inside, stacked, holding)
      (c) object-environment: relations to workspace (e.g. on-table, clear, arm-empty, reachable)

Aim for at least 2-3 predicates per category (typically 8-15 predicates total).
Do NOT include any :action blocks.

Respond with ONLY the PDDL domain block, e.g.:

(define (domain tabletop)
  (:requirements :strips :typing)
  (:types
    robot - object
    movable surface - object
    block cup container - movable
    table - surface
  )
  (:predicates
    ;; Object-self
    (opened ?c - container)
    (closed ?c - container)
    (empty ?c - container)
    (upright ?m - movable)
    ;; Object-object
    (on ?x - movable ?y - object)
    (inside ?x - movable ?c - container)
    (holding ?r - robot ?m - movable)
    (stacked ?x - movable ?y - movable)
    ;; Object-environment
    (on-table ?m - movable ?t - surface)
    (clear ?x - object)
    (arm-empty ?r - robot)
    (reachable ?r - robot ?x - object)
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

def _extract_balanced_sexp(text: str, opener: str = "(define") -> str | None:
    """
    Extract the first balanced S-expression starting with *opener*.

    Uses parenthesis-depth tracking so that any "thinking" text the
    model emits before, after, or *inside* the block is stripped.
    Quoted strings (``"…"``) are treated as opaque — parentheses inside
    them are ignored.
    """
    start = text.find(opener)
    if start < 0:
        return None

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
                    return text[start : i + 1]
        prev_ch = ch

    # Unbalanced — return None; caller handles gracefully
    return None


# Regex for lines that are clearly natural-language prose, not PDDL.
_PROSE_LINE_RE = re.compile(
    r"^[A-Z].*[.!?]$"    # Starts with capital, ends with sentence punctuation
    r"|`"                 # Contains markdown backticks
    r"|^(Wait|Hmm|Let|Note|Actually|I think|I'll|So |Also |Oh )",
    re.IGNORECASE,
)


def _sanitize_pddl_block(text: str) -> str:
    """
    Remove natural-language "thinking" lines interleaved inside PDDL.

    Keeps: empty lines, PDDL comments (``; …``), and lines that contain
    typical PDDL characters (parentheses, ``:``, ``?``) or look like
    bare type/predicate identifiers (e.g. ``block cup - movable``).
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Always keep empty lines and comments
        if not stripped or stripped.startswith(";"):
            cleaned.append(line)
            continue
        # Keep lines with PDDL structural characters
        if any(ch in stripped for ch in "():?"):
            cleaned.append(line)
            continue
        # Keep bare identifier lines (valid in :types blocks)
        # e.g. "block cup - movable" or "movable table - object"
        if re.match(r"^[\w][\w\s-]*$", stripped):
            cleaned.append(line)
            continue
        # Skip lines that look like English prose
        if _PROSE_LINE_RE.search(stripped):
            continue
        # Default: keep (be conservative)
        cleaned.append(line)
    return "\n".join(cleaned)


def parse_vlm_domain_response(response_text: str) -> str:
    """
    Extract the PDDL domain block from the VLM response.

    Handles models that emit "thinking" text (chain-of-thought) before,
    after, or even interleaved with the PDDL block.  Uses balanced
    parenthesis extraction so only the ``(define …)`` S-expression is
    returned.

    Validation / conversion to pyPPDDL objects is left to the caller
    (``domain_genesis.py``).
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

    # Try balanced extraction of (define …)
    sexp = _extract_balanced_sexp(text, "(define")
    if sexp is not None:
        return _sanitize_pddl_block(sexp)

    # Fallback: return as-is; let caller deal with parse errors
    return _sanitize_pddl_block(text)
