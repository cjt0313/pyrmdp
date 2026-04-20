"""VLM-based logical state estimation for video frames.

Uses the same OpenAI multimodal API pattern as
``pyrmdp.synthesis.domain_genesis._build_default_vlm_fn``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

VlmFn = Callable[[str, str, List[Dict[str, str]]], str]

# ─────────────────────────────────────────────────────────────
#  VLM callable builder
# ─────────────────────────────────────────────────────────────

def build_vlm_fn(config=None) -> VlmFn:
    """Build a VLM callable ``(system, user_text, images) → str``.

    Mirrors ``domain_genesis._build_default_vlm_fn`` so the same
    ``llm.yaml`` / ``LLMConfig`` drives both pipelines.
    """
    if config is None:
        from pyrmdp.synthesis.llm_config import load_config
        config = load_config()

    from openai import OpenAI

    client_kw: dict = {"api_key": config.api_key}
    if config.base_url:
        client_kw["base_url"] = config.base_url
    if config.timeout:
        client_kw["timeout"] = config.timeout
    if config.max_retries:
        client_kw["max_retries"] = config.max_retries

    client = OpenAI(**client_kw)

    def _call(system: str, user_text: str, images: List[Dict[str, str]]) -> str:
        content: list = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
            })
        content.append({"type": "text", "text": user_text})

        resp = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=config.max_tokens,
        )
        return resp.choices[0].message.content

    return _call


def _encode_frame(frame: np.ndarray) -> Dict[str, str]:
    """Encode an RGB numpy array as a base64 JPEG dict."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"mime": "image/jpeg", "base64": b64}


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from VLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No JSON object found in VLM response: {text[:200]}")


# ─────────────────────────────────────────────────────────────
#  Bootstrap: discover grounded predicates from first frame
# ─────────────────────────────────────────────────────────────

_GROUNDING_SYSTEM = (
    "You are a PDDL grounding assistant. Given one or more images of a "
    "robot scene captured from different camera viewpoints, a task "
    "instruction, and a list of lifted predicate signatures, produce a "
    "list of grounded predicates for ONLY the primary objects mentioned "
    "in the task instruction. IGNORE distractor objects that are not "
    "relevant to the task. Each grounded predicate replaces typed "
    "variables with the concrete object names visible in the scene."
)

_GROUNDING_USER = """\
Task: {instruction}

Domain predicate signatures (PDDL):
{signatures}

You are given {num_views} camera view(s) of the same scene at the same \
time step. Use ALL views together to identify objects and their relations.

Identify ONLY the objects directly mentioned in or required by the task \
instruction above. Ground EVERY predicate signature above with those \
task-relevant objects — even predicates that are currently false. The \
predicates will be tracked over time, so we need groundings for all of them.

Return a JSON object:
{{"grounded_predicates": ["pred(obj1, obj2)", ...]}}

Rules:
- Only ground predicates for task-relevant objects. Ignore distractors.
- Use short, descriptive object names visible in the scene (e.g., \
"robot", "white_toy", "drawer").
- You MUST produce at least one grounding for EVERY predicate signature \
listed above. If a predicate could apply to a task-relevant object at any \
point during the task, include it."""


def discover_grounding(
    vlm_fn: VlmFn,
    first_frame: np.ndarray | List[np.ndarray],
    predicate_signatures: List[str],
    language_instruction: str,
) -> List[str]:
    """Bootstrap: VLM discovers grounded predicates from frame 0.

    Parameters
    ----------
    vlm_fn
        Multimodal callable ``(system, user_text, images) → str``.
    first_frame
        Single RGB image (H, W, 3) or list of images from multiple cameras.
    predicate_signatures
        PDDL-style sigs, e.g. ``["holding ?r - robot ?x - movable", ...]``.
    language_instruction
        Task description from the HDF5 file.

    Returns
    -------
    list[str]
        Grounded predicate strings, e.g. ``["holding(robot, white_toy)", ...]``.
    """
    frames = first_frame if isinstance(first_frame, list) else [first_frame]
    sig_block = "\n".join(f"  ({s})" for s in predicate_signatures)
    user = _GROUNDING_USER.format(
        instruction=language_instruction,
        signatures=sig_block,
        num_views=len(frames),
    )
    imgs = [_encode_frame(f) for f in frames]
    raw = vlm_fn(_GROUNDING_SYSTEM, user, imgs)
    logger.debug("Bootstrap VLM raw response:\n%s", raw)
    data = _extract_json(raw)
    preds = data.get("grounded_predicates", [])
    logger.info("Bootstrap discovered %d grounded predicates", len(preds))
    return preds


# ─────────────────────────────────────────────────────────────
#  Per-frame state estimation
# ─────────────────────────────────────────────────────────────

_STATE_SYSTEM = (
    "You are a logical state estimator. Given one or more images of the "
    "same scene from different camera viewpoints and a list of boolean "
    "predicates, evaluate which predicates are True and which are False "
    "in the current scene. Use ALL views together to make your judgment. "
    "Return ONLY a JSON object mapping each predicate to true or false."
)

_STATE_USER = """\
You are given {num_views} camera view(s) of the same scene at the same \
time step. Evaluate each predicate using ALL views together:
{pred_list}

For each grounded predicate, return true if the relation holds for those \
specific objects, or false if it does not. Do not omit any predicate.

Return a strict JSON mapping:
{{"pred1(obj)": true, "pred2(obj)": false, ...}}
Every predicate above MUST appear in your response."""


def estimate_frame_state(
    vlm_fn: VlmFn,
    frame: np.ndarray | List[np.ndarray],
    grounded_predicates: List[str],
) -> Dict[str, bool]:
    """Estimate the logical state of a single time step.

    Parameters
    ----------
    frame
        Single RGB image (H, W, 3) or list of images from multiple cameras.
    """
    frames = frame if isinstance(frame, list) else [frame]
    pred_list = "\n".join(f"  - {p}" for p in grounded_predicates)
    user = _STATE_USER.format(pred_list=pred_list, num_views=len(frames))
    imgs = [_encode_frame(f) for f in frames]

    raw = vlm_fn(_STATE_SYSTEM, user, imgs)
    data = _extract_json(raw)

    state: Dict[str, bool] = {}
    for p in grounded_predicates:
        if p in data:
            state[p] = bool(data[p])
        else:
            for k, v in data.items():
                if k.replace(" ", "") == p.replace(" ", ""):
                    state[p] = bool(v)
                    break
            else:
                logger.warning("Predicate '%s' missing from VLM response", p)
                state[p] = False

    return state


def estimate_trajectory(
    vlm_fn: VlmFn,
    frames: List[np.ndarray] | List[List[np.ndarray]],
    grounded_predicates: List[str],
    max_workers: int = 4,
) -> List[Dict[str, bool]]:
    """Estimate states for all time steps using a thread pool.

    Parameters
    ----------
    frames
        Either a list of single images (one camera) or a list of
        image-lists (multiple cameras per time step).

    Returns a list of state dicts in frame order.
    """
    results: List[Optional[Dict[str, bool]]] = [None] * len(frames)

    def _work(idx: int) -> tuple:
        return idx, estimate_frame_state(vlm_fn, frames[idx], grounded_predicates)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, i): i for i in range(len(frames))}
        done = 0
        for fut in as_completed(futures):
            idx, state = fut.result()
            results[idx] = state
            done += 1
            if done % 5 == 0 or done == len(frames):
                logger.info("  VLM progress: %d/%d frames", done, len(frames))

    return results  # type: ignore[return-value]
