"""
FastAPI server for the Online Update Experiment.

Run with::

    uvicorn pyrmdp.online_experiment.backend.server:app --host 0.0.0.0 --port 8000

Environment variables (read at startup):

    PYRMDP_DOMAIN_PATH   path to the PPDDL domain (default: pipeline_output/robustified.ppddl)
    PYRMDP_EPS_PHYS      LCB threshold for severing edges (default: 0.35)
    PYRMDP_EPS_SPECTRAL  Wasserstein convergence threshold (default: 0.02)
    PYRMDP_OUTPUT_DIR    where to dump experiment artifacts (default: ./online_output)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .online_updater import FailureReport, OnlineUpdater
from .state_manager import GraphManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("online_experiment.server")


# ════════════════════════════════════════════════════════════════════
#  Pydantic payloads
# ════════════════════════════════════════════════════════════════════

class FailureType(str, Enum):
    kinematic = "kinematic"
    known_state = "known_state"
    new_state = "new_state"
    new_predicate = "new_predicate"
    new_state_from_existing = "new_state_from_existing"


class SuccessPayload(BaseModel):
    action_name: str


class UnchangedPayload(BaseModel):
    action_name: str


class FailurePayload(BaseModel):
    action_name: str
    failure_type: FailureType
    image_b64: Optional[str] = Field(None, description="Base64 PNG/JPEG")
    expert_feedback: Optional[str] = None
    # new_predicate fields
    predicate_name: Optional[str] = None
    parameter_types: Optional[List[str]] = None
    # new_state_from_existing fields
    true_predicates: Optional[List[str]] = None
    false_predicates: Optional[List[str]] = None
    state_label: Optional[str] = None


class BaselineSkillPayload(BaseModel):
    operator: str
    note: Optional[str] = None
    human_id: str = "H2"


# ════════════════════════════════════════════════════════════════════
#  App + singletons
# ════════════════════════════════════════════════════════════════════

app = FastAPI(title="pyrmdp Online Update Experiment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    domain_path = os.environ.get(
        "PYRMDP_DOMAIN_PATH",
        "pipeline_output/robustified.ppddl",
    )
    eps_phys = float(os.environ.get("PYRMDP_EPS_PHYS", "0.35"))
    eps_spec = float(os.environ.get("PYRMDP_EPS_SPECTRAL", "0.02"))
    out_dir = os.environ.get("PYRMDP_OUTPUT_DIR", "./online_output")
    if not Path(domain_path).exists():
        raise RuntimeError(
            f"PPDDL domain not found: {domain_path}. "
            "Run the offline pipeline first or set PYRMDP_DOMAIN_PATH."
        )
    manager = GraphManager(
        domain_path,
        epsilon_phys=eps_phys,
        epsilon_spectral=eps_spec,
        output_dir=out_dir,
    )
    updater = OnlineUpdater(manager)
    app.state.manager = manager
    app.state.updater = updater
    app.state.baseline_skills = []  # List[Dict[str, Any]]
    logger.info(
        "Online experiment ready  |  ε_phys=%.2f  ε_spec=%.3f  domain=%s",
        eps_phys, eps_spec, domain_path,
    )


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _mgr() -> GraphManager:
    return app.state.manager


def _upd() -> OnlineUpdater:
    return app.state.updater


# ════════════════════════════════════════════════════════════════════
#  Endpoints
# ════════════════════════════════════════════════════════════════════

@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "pyrmdp-online-experiment", "status": "ok"}


@app.get("/get_status")
def get_status() -> Dict[str, Any]:
    return _mgr().status()


@app.get("/get_next_action")
def get_next_action() -> Dict[str, Optional[str]]:
    return {"action_name": _mgr().current_action}


@app.get("/get_event_log")
def get_event_log(tail: int = 100) -> Dict[str, Any]:
    return {"events": _mgr().event_log_tail(tail)}


@app.get("/get_graph")
def get_graph() -> Dict[str, Any]:
    """Live abstract transition graph (vis.js-friendly JSON)."""
    return _mgr().serialize_graph()


@app.get("/get_domain_info")
def get_domain_info() -> Dict[str, Any]:
    """Types, predicate signatures and existing abstract states — used by
    the expert-guided failure UI to populate dropdowns."""
    return _mgr().domain_info()


@app.post("/report_success")
def report_success(payload: SuccessPayload) -> Dict[str, Any]:
    return _upd().report_success(payload.action_name)


@app.post("/report_unchanged")
def report_unchanged(payload: UnchangedPayload) -> Dict[str, Any]:
    return _upd().report_unchanged(payload.action_name)


@app.post("/report_failure")
def report_failure(payload: FailurePayload) -> Dict[str, Any]:
    try:
        return _upd().report_failure(
            FailureReport(
                action_name=payload.action_name,
                failure_type=payload.failure_type.value,
                image_b64=payload.image_b64,
                expert_feedback=payload.expert_feedback,
                predicate_name=payload.predicate_name,
                parameter_types=payload.parameter_types,
                true_predicates=payload.true_predicates,
                false_predicates=payload.false_predicates,
                state_label=payload.state_label,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/re_synthesize")
def re_synthesize() -> Dict[str, Any]:
    """Force an MSCA re-synthesis round (useful from the dashboard)."""
    return _upd().re_synthesize()


# ── Human 2 (Heuristic Baseline) log ─────────────────────────────

@app.post("/baseline/add_skill")
def baseline_add_skill(payload: BaselineSkillPayload) -> Dict[str, Any]:
    import time
    entry = {
        "t": time.time(),
        "operator": payload.operator,
        "note": payload.note,
        "human_id": payload.human_id,
    }
    app.state.baseline_skills.append(entry)
    # Persist to disk for later comparison
    out = _mgr().output_dir / "baseline_skills.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        import json
        fh.write(json.dumps(entry) + "\n")
    return {"ok": True, "count": len(app.state.baseline_skills)}


@app.get("/baseline/list")
def baseline_list() -> Dict[str, Any]:
    return {"skills": app.state.baseline_skills}
