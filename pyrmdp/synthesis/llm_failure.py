"""
Step 1: LLM Failure Hallucination on Base Domain

For each operator in a parsed PDDL/PPDDL domain, query an LLM to hallucinate
a physically plausible "worse effect" (failure mode). If the failure introduces
novel predicates or types, the domain is updated accordingly.

Uses pyPPDDL's data model (Domain, ActionSchema, Effect, Predicate, TypedParam).
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# pyPPDDL data model
try:
    from pyppddl.ppddl.parser import (
        ActionSchema,
        Domain,
        Effect,
        Predicate,
        TypedParam,
    )
except ImportError:
    raise ImportError(
        "pyPPDDL is required. Install it with: "
        "pip install -e /path/to/pyPPDDL"
    )

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  LLM Interface
# ════════════════════════════════════════════════════════════════════

@dataclass
class FailureHallucinationResult:
    """Result of hallucinating a failure mode for one action."""
    action_name: str
    failure_effects: Effect
    new_predicates: List[Predicate] = field(default_factory=list)
    new_types: Dict[str, str] = field(default_factory=dict)  # child -> parent
    raw_llm_response: str = ""


def _format_action_for_prompt(action: ActionSchema, domain: Domain) -> str:
    """Serialize an ActionSchema into a human-readable string for the LLM prompt."""
    params_str = " ".join(
        f"{p.name} - {p.type}" for p in action.parameters
    )
    
    # Format precondition S-expression
    def _sexp_to_str(sexp: Any) -> str:
        if sexp is None:
            return "()"
        if isinstance(sexp, list):
            inner = " ".join(_sexp_to_str(x) for x in sexp)
            return f"({inner})"
        return str(sexp)
    
    precond_str = _sexp_to_str(action.precondition)
    
    effects_str = []
    for eff in action.effects:
        parts = []
        if eff.prob < 1.0:
            parts.append(f"  probability: {eff.prob}")
        if eff.add_predicates:
            parts.append(f"  add: {eff.add_predicates}")
        if eff.del_predicates:
            parts.append(f"  del: {eff.del_predicates}")
        if eff.numeric_effects:
            parts.append(f"  numeric: {eff.numeric_effects}")
        effects_str.append("\n".join(parts))
    
    # Domain context
    pred_names = [p.name for p in domain.predicates]
    type_names = list(domain.types.keys())
    
    return (
        f"Domain predicates: {pred_names}\n"
        f"Domain types: {type_names}\n"
        f"Action: {action.name}\n"
        f"Parameters: ({params_str})\n"
        f"Precondition: {precond_str}\n"
        f"Effects:\n" + "\n---\n".join(effects_str)
    )


def _build_failure_prompt(action_desc: str) -> str:
    """Build the LLM prompt for failure hallucination."""
    return f"""You are a robotics domain expert. Given the following PDDL operator, 
hallucinate ONE physically plausible "worse effect" (failure mode) that could occur
when executing this action. The failure should be different from "no change" (unchanged).

{action_desc}

Requirements:
1. The failure must be physically plausible for a robotics scenario.
2. If the failure introduces a new physical state not captured by existing predicates,
   propose NEW predicates (with typed parameters) and/or new types.
3. The failure effect should be expressible as PDDL add/del effects.
4. Keep it minimal — one failure mode, not multiple.

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


def _parse_llm_response(
    response_text: str, action: ActionSchema
) -> Optional[FailureHallucinationResult]:
    """Parse the LLM JSON response into a FailureHallucinationResult."""
    try:
        # Extract JSON from response (handle markdown code fences)
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning(f"No JSON found in LLM response for {action.name}")
            return None
        
        data = json.loads(text[start:end])
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response for {action.name}: {e}")
        return None
    
    # Build the failure Effect
    failure_eff = Effect(prob=1.0)
    
    for pred in data.get("failure_add", []):
        failure_eff.add_predicates.append(tuple(pred))
    
    for pred in data.get("failure_del", []):
        failure_eff.del_predicates.append(tuple(pred))
    
    for num_eff in data.get("failure_numeric", []):
        if len(num_eff) >= 3:
            failure_eff.numeric_effects.append(tuple(num_eff))
    
    # Parse new predicates
    new_predicates = []
    for p_data in data.get("new_predicates", []):
        params = [
            TypedParam(name=pp["name"], type=pp.get("type", "object"))
            for pp in p_data.get("parameters", [])
        ]
        new_predicates.append(Predicate(name=p_data["name"], parameters=params))
    
    # Parse new types
    new_types = data.get("new_types", {})
    
    return FailureHallucinationResult(
        action_name=action.name,
        failure_effects=failure_eff,
        new_predicates=new_predicates,
        new_types=new_types,
        raw_llm_response=response_text,
    )


def _make_default_llm_fn() -> Any:
    """Build the default LLM callable from llm.yaml / env-vars."""
    from .llm_config import build_llm_fn
    return build_llm_fn()


# ════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ════════════════════════════════════════════════════════════════════

def hallucinate_failures(
    domain: Domain,
    *,
    llm_fn: Optional[Any] = None,
    failure_prob: float = 0.1,
) -> Tuple[Domain, List[FailureHallucinationResult]]:
    """
    For each action in *domain*, hallucinate a failure mode via LLM.
    
    Returns a new Domain with:
      - Each action's effects updated with a failure branch
      - New predicates/types added as needed
    
    Parameters
    ----------
    domain : Domain
        The parsed pyPPDDL Domain object.
    llm_fn : callable, optional
        Custom function(prompt: str) -> str.  If None, builds one from
        ``llm.yaml`` / env-vars via :func:`llm_config.build_llm_fn`.
    failure_prob : float
        Probability assigned to the failure branch (default 0.1).
        Existing effect probabilities are rescaled to (1 - failure_prob).
    
    Returns
    -------
    (augmented_domain, results) : Tuple[Domain, List[FailureHallucinationResult]]
    """
    query_fn = llm_fn or _make_default_llm_fn()
    augmented = deepcopy(domain)
    results: List[FailureHallucinationResult] = []
    
    for action in augmented.actions:
        logger.info(f"Hallucinating failure for action: {action.name}")
        
        action_desc = _format_action_for_prompt(action, augmented)
        prompt = _build_failure_prompt(action_desc)
        
        try:
            raw_response = query_fn(prompt)
        except Exception as e:
            logger.error(f"LLM query failed for {action.name}: {e}")
            continue
        
        result = _parse_llm_response(raw_response, action)
        if result is None:
            logger.warning(f"Could not parse failure for {action.name}, skipping.")
            continue
        
        results.append(result)
        
        # ── Update domain with new predicates/types ──
        existing_pred_names = {p.name for p in augmented.predicates}
        for new_pred in result.new_predicates:
            if new_pred.name not in existing_pred_names:
                augmented.predicates.append(new_pred)
                existing_pred_names.add(new_pred.name)
                logger.info(f"  Added new predicate: {new_pred.name}")
        
        for child_type, parent_type in result.new_types.items():
            if child_type not in augmented.types:
                augmented.types[child_type] = parent_type
                logger.info(f"  Added new type: {child_type} - {parent_type}")
        
        # ── Inject failure branch into action effects ──
        # Scale existing effects to (1 - failure_prob)
        scale = 1.0 - failure_prob
        for eff in action.effects:
            eff.prob *= scale
        
        # Add failure branch
        result.failure_effects.prob = failure_prob
        action.effects.append(result.failure_effects)
    
    return augmented, results
