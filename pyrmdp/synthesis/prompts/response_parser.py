"""
Shared response-parsing utilities for LLM prompt modules.

Every LLM call in the synthesis pipeline returns free-form text that
*should* contain JSON.  In practice, models wrap JSON in markdown code
fences, add trailing commas, or prepend/append explanatory prose.

This module centralises the extraction + sanitisation logic so that
each prompt module's ``parse_*_response()`` can focus on *semantic*
validation rather than re-implementing the same regex boilerplate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> Optional[Any]:
    """
    Extract and parse the first JSON object (``{…}``) from LLM output.

    Processing steps
    ----------------
    1. Strip markdown code fences (````json … ````).
    2. Locate the first ``{`` and last ``}`` to isolate the JSON body.
    3. Remove trailing commas before ``}`` or ``]`` (common LLM error).
    4. Parse with :func:`json.loads`.

    Parameters
    ----------
    text : str
        Raw LLM response text.

    Returns
    -------
    dict | list | None
        Parsed JSON value, or *None* if extraction/parsing fails.
    """
    cleaned = text.strip()

    # ── 1. Strip markdown fences ──────────────────────────────────
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    # ── 2. Isolate JSON body ──────────────────────────────────────
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        logger.warning("No JSON object found in LLM response")
        logger.debug("  Raw text (first 500 chars):\n%s", text[:500])
        return None

    json_str = cleaned[start:end]

    # ── 3. Fix trailing commas ────────────────────────────────────
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # ── 4. Parse ──────────────────────────────────────────────────
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s", exc)
        logger.debug("  Attempted JSON:\n%s", json_str[:500])
        return None
