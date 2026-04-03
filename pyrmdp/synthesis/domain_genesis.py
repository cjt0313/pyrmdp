"""
Step 0: Domain Genesis — RGB + Task Description → PDDL Domain

Two-phase pipeline that generates a complete initial PDDL domain from
scratch, before the robustification pipeline (Steps 1–6) runs:

  **Phase A (VLM):**  One or more RGB images of the scene are sent to a
  Vision-Language Model.  The VLM returns the `:types` and `:predicates`
  section — always including a ``robot`` type.

  **Phase B (LLM):**  The generated types/predicates plus one or more
  natural-language task descriptions are sent to an LLM.  It returns the
  `:action` blocks (deterministic, STRIPS-compatible).

The two phases are composed by :func:`generate_initial_domain`, which
returns either a raw PDDL string or a parsed ``pyPPDDL.Domain`` object
(if pyPPDDL is available).

Usage
-----
>>> from pyrmdp.synthesis.domain_genesis import generate_initial_domain
>>> pddl_str = generate_initial_domain(
...     image_paths=["scene.png"],
...     task_descriptions=["Pick up the red block and place it on the blue plate."],
... )
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .prompts.vlm_domain_prompt import (
    build_vlm_domain_prompt,
    parse_vlm_domain_response,
)
from .prompts.llm_operator_prompt import (
    build_llm_operator_prompt,
    parse_llm_operator_response,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Image helpers
# ════════════════════════════════════════════════════════════════════

def _load_image_as_base64(path: Union[str, Path]) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_mime(path: Union[str, Path]) -> str:
    """Guess MIME type from extension."""
    ext = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


# ════════════════════════════════════════════════════════════════════
#  VLM / LLM call builders  (provider-agnostic)
# ════════════════════════════════════════════════════════════════════

def _build_default_vlm_fn() -> Callable:
    """Build a VLM callable using the shared LLMConfig (must support vision)."""
    from .llm_config import load_config

    config = load_config()

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install 'pyrmdp[llm]'")

    if not config.api_key:
        raise EnvironmentError(
            "No API key found.  Set OPENAI_API_KEY or configure llm.yaml."
        )

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    def _call(
        system: str,
        user_text: str,
        images: List[Dict[str, str]],  # [{"mime": …, "base64": …}, …]
    ) -> str:
        """Send a vision-capable chat completion request."""
        # Build the multimodal user message
        content: list = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['mime']};base64,{img['base64']}",
                },
            })
        content.append({"type": "text", "text": user_text})

        resp = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return resp.choices[0].message.content

    return _call


def _build_default_llm_fn() -> Callable:
    """Build a text-only LLM callable from shared config."""
    from .llm_config import build_llm_fn
    return build_llm_fn()


# ════════════════════════════════════════════════════════════════════
#  PDDL Assembly
# ════════════════════════════════════════════════════════════════════

def _assemble_domain(
    domain_fragment: str,
    operator_fragment: str,
) -> str:
    """
    Merge the VLM's types+predicates block with the LLM's action blocks
    into a single well-formed PDDL domain string.

    If the VLM already returned a full ``(define (domain …) …)`` block,
    the actions are inserted before the closing parenthesis.  Otherwise
    a new ``(define …)`` wrapper is created.
    """
    domain_fragment = domain_fragment.strip()
    operator_fragment = operator_fragment.strip()

    if domain_fragment.startswith("(define"):
        # Insert actions before the final closing paren
        # Find the last ')' that closes the (define …)
        depth = 0
        last_close = -1
        for i, ch in enumerate(domain_fragment):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    last_close = i
                    break

        if last_close > 0:
            return (
                domain_fragment[:last_close]
                + "\n\n  ;; ── Operators (auto-generated) ──\n"
                + _indent_block(operator_fragment, level=1)
                + "\n"
                + domain_fragment[last_close:]
            )

    # Fallback: wrap everything
    return (
        "(define (domain auto-generated)\n"
        "  (:requirements :strips :typing)\n\n"
        f"  {domain_fragment}\n\n"
        "  ;; ── Operators (auto-generated) ──\n"
        f"{_indent_block(operator_fragment, level=1)}\n"
        ")\n"
    )


def _indent_block(text: str, level: int = 1) -> str:
    """Indent every line of *text* by *level* × 2 spaces."""
    prefix = "  " * level
    return "\n".join(prefix + line for line in text.split("\n"))


# ════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ════════════════════════════════════════════════════════════════════

def generate_initial_domain(
    *,
    image_paths: List[Union[str, Path]],
    task_descriptions: List[str],
    scene_description: str = "",
    domain_name_hint: Optional[str] = None,
    vlm_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    return_parsed: bool = False,
) -> Union[str, Any]:
    """
    Generate a complete PDDL domain from RGB observations and task text.

    Parameters
    ----------
    image_paths : list[str | Path]
        Paths to one or more RGB images of the scene.
    task_descriptions : list[str]
        Natural-language sentences describing the robot's task / policy.
        Examples: ``["Stack all blocks on the tray.",
                      "Avoid placing anything on the red zone."]``
    scene_description : str
        Optional extra text for the VLM (e.g. "A Franka robot on a
        tabletop with coloured blocks").
    domain_name_hint : str, optional
        Suggested PDDL domain name.
    vlm_fn : callable, optional
        ``fn(system, user_text, images) → str``.  If None, built from
        ``llm.yaml`` config (must be a vision-capable model).
    llm_fn : callable, optional
        ``fn(prompt) → str``.  If None, built from ``llm.yaml`` config.
    return_parsed : bool
        If True and pyPPDDL is available, return a parsed ``Domain``
        object instead of a raw string.

    Returns
    -------
    str or Domain
        The assembled PDDL domain string, or a ``pyPPDDL.Domain``
        object if ``return_parsed=True``.
    """
    # ── Phase A: VLM → types + predicates ──────────────────────────

    logger.info("Step 0a: Querying VLM for object types & predicates …")

    vlm_call = vlm_fn or _build_default_vlm_fn()
    prompt_a = build_vlm_domain_prompt(
        scene_description=scene_description,
        domain_name_hint=domain_name_hint,
    )

    images = [
        {"mime": _image_mime(p), "base64": _load_image_as_base64(p)}
        for p in image_paths
    ]

    raw_vlm = vlm_call(prompt_a["system"], prompt_a["user"], images)
    domain_fragment = parse_vlm_domain_response(raw_vlm)

    logger.info("  VLM returned domain fragment (%d chars)", len(domain_fragment))
    logger.debug("  Domain fragment:\n%s", domain_fragment)

    # ── Phase B: LLM → operators ──────────────────────────────────

    logger.info("Step 0b: Querying LLM for operators …")

    llm_call = llm_fn or _build_default_llm_fn()
    prompt_b = build_llm_operator_prompt(
        domain_fragment=domain_fragment,
        task_descriptions=task_descriptions,
    )

    # The shared build_llm_fn returns fn(prompt)->str (single string).
    # We concatenate system + user for the text-only call.
    full_prompt = prompt_b["system"] + "\n\n" + prompt_b["user"]
    raw_llm = llm_call(full_prompt)
    operator_fragment = parse_llm_operator_response(raw_llm)

    logger.info("  LLM returned operators (%d chars)", len(operator_fragment))
    logger.debug("  Operator fragment:\n%s", operator_fragment)

    # ── Assemble ──────────────────────────────────────────────────

    pddl_str = _assemble_domain(domain_fragment, operator_fragment)
    logger.info("Step 0: Domain genesis complete (%d chars)", len(pddl_str))

    if return_parsed:
        try:
            from pyppddl.ppddl.parser import load_domain_from_string
            return load_domain_from_string(pddl_str)
        except ImportError:
            logger.warning(
                "pyPPDDL not installed — returning raw PDDL string instead."
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse generated PDDL: %s — returning raw string.", exc
            )

    return pddl_str
