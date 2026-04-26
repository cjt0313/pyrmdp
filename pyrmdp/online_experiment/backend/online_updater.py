"""
Online Updater
==============

Routes a failure/success report through the mathematical machinery:

    ┌─ success ─────────── Dirichlet(n_success ↑)
    │
    ┌─ unchanged ───────── Dirichlet(n_unchanged ↑)
    │
report_failure
    ├─ kinematic ────────── force P=0, sever edge, tabu
    ├─ known_state ──────── Dirichlet(n_failure ↑); if LCB < ε_phys → sever
    └─ new_state ────────── parse image/expert text → add Trap State

Whenever the graph topology is mutated (edge severed or node added) the
updater recomputes the Wasserstein spectral distance.  If
``Δ_W > ε_spectral`` it triggers a *background* MSCA re-synthesis via
:class:`IterativeDomainSynthesizer.delta_minimize` path and waits until
``Δ_W`` drops below ``ε_spectral``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .state_manager import DirichletCounts, GraphManager

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Input payloads
# ════════════════════════════════════════════════════════════════════

FAILURE_TYPES = {"kinematic", "known_state", "new_state",
                 "new_predicate", "new_state_from_existing"}


@dataclass
class FailureReport:
    action_name: str
    failure_type: str                 # one of FAILURE_TYPES
    image_b64: Optional[str] = None
    expert_feedback: Optional[str] = None
    # Expert-guided new-predicate fields
    predicate_name: Optional[str] = None
    parameter_types: Optional[List[str]] = None
    # Expert-guided new-state-from-existing fields
    true_predicates: Optional[List[str]] = None
    false_predicates: Optional[List[str]] = None
    state_label: Optional[str] = None

    def validate(self) -> None:
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(
                f"failure_type must be one of {FAILURE_TYPES}, "
                f"got {self.failure_type!r}"
            )


# ════════════════════════════════════════════════════════════════════
#  Updater
# ════════════════════════════════════════════════════════════════════

class OnlineUpdater:
    """Mutate a :class:`GraphManager` in response to outcome reports."""

    def __init__(
        self,
        manager: GraphManager,
        *,
        vlm_fn=None,
    ) -> None:
        self.mgr = manager
        self.vlm_fn = vlm_fn            # optional callable(image_b64, text) → [pred_str]
        self._resync_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def report_success(self, action_name: str) -> Dict[str, Any]:
        counts = self.mgr.update_dirichlet(action_name, success=1.0)
        self.mgr.log_event("success", action=action_name,
                           lcb=counts.p_success_lcb())
        self.mgr.advance_action()
        return {"ok": True, "lcb": counts.p_success_lcb(),
                "next_action": self.mgr.current_action}

    def report_unchanged(self, action_name: str) -> Dict[str, Any]:
        counts = self.mgr.update_dirichlet(action_name, unchanged=1.0)
        self.mgr.log_event("unchanged", action=action_name,
                           lcb=counts.p_success_lcb())
        self.mgr.advance_action()
        return {"ok": True, "lcb": counts.p_success_lcb(),
                "next_action": self.mgr.current_action}

    def report_failure(self, report: FailureReport) -> Dict[str, Any]:
        report.validate()
        action = report.action_name
        topology_changed = False
        result: Dict[str, Any] = {"action": action,
                                  "failure_type": report.failure_type}

        if report.failure_type == "kinematic":
            counts = self.mgr.update_dirichlet(action, kinematic_kill=True)
            removed = self.mgr.sever_edges_for(
                action, reason="kinematic", lcb=0.0,
            )
            topology_changed = bool(removed)
            result["severed_edges"] = removed
            result["lcb"] = counts.p_success_lcb()

        elif report.failure_type == "known_state":
            counts = self.mgr.update_dirichlet(action, failure=1.0)
            lcb = counts.p_success_lcb()
            result["lcb"] = lcb
            if lcb < self.mgr.epsilon_phys:
                removed = self.mgr.sever_edges_for(
                    action, reason="lcb<eps_phys", lcb=lcb,
                )
                topology_changed = bool(removed)
                result["severed_edges"] = removed

        elif report.failure_type == "new_state":
            counts = self.mgr.update_dirichlet(action, failure=1.0)
            predicates, label = self._extract_new_predicates(
                report.image_b64, report.expert_feedback,
            )
            trap_id = self.mgr.add_trap_state(
                label=label,
                predicates=predicates,
                from_action=action,
            )
            topology_changed = True
            result["trap_state"] = trap_id
            result["predicates"] = predicates
            result["lcb"] = counts.p_success_lcb()

        elif report.failure_type == "new_state_from_existing":
            # Expert picks true/false assignments from existing predicates.
            counts = self.mgr.update_dirichlet(action, failure=1.0)
            true_preds = list(report.true_predicates or [])
            false_preds = list(report.false_predicates or [])
            if not (true_preds or false_preds):
                raise ValueError(
                    "new_state_from_existing requires at least one "
                    "true or false predicate",
                )
            nid = self.mgr.add_abstract_state(
                true_predicates=true_preds,
                false_predicates=false_preds,
                from_action=action,
                label=report.state_label,
            )
            topology_changed = True
            result["trap_state"] = nid
            result["true_predicates"] = true_preds
            result["false_predicates"] = false_preds
            result["lcb"] = counts.p_success_lcb()

        elif report.failure_type == "new_predicate":
            # Expert declares a brand-new predicate symbol → add to domain,
            # rebuild graph, then trigger MSCA reflex.
            if not report.predicate_name:
                raise ValueError("predicate_name is required")
            counts = self.mgr.update_dirichlet(action, failure=1.0)
            added = self.mgr.add_predicate(
                report.predicate_name.strip(),
                list(report.parameter_types or []),
            )
            stats = self.mgr.rebuild_graph()
            topology_changed = True
            result["predicate"] = added
            result["rebuild"] = stats
            result["lcb"] = counts.p_success_lcb()

        self.mgr.log_event(
            "failure",
            action=action,
            failure_type=report.failure_type,
            **{k: v for k, v in result.items()
               if k not in {"action", "failure_type"}},
        )

        if topology_changed:
            delta = self.mgr.refresh_spectral()
            result["spectral_delta"] = delta
            if delta > self.mgr.epsilon_spectral:
                # Fire MSCA in-thread for the skeleton; the server can move
                # this into a BackgroundTasks queue if latency matters.
                result["resynthesis"] = self.re_synthesize()

        self.mgr.advance_action()
        result["next_action"] = self.mgr.current_action
        return result

    # ──────────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────────

    def _extract_new_predicates(
        self,
        image_b64: Optional[str],
        expert_text: Optional[str],
    ) -> tuple[list[str], str]:
        """Attempt VLM extraction; fall back to whitespace-split expert text."""
        preds: List[str] = []
        label = "unknown"
        if self.vlm_fn is not None and image_b64:
            try:
                preds = list(self.vlm_fn(image_b64, expert_text))
                label = "vlm"
            except Exception as exc:
                logger.warning("VLM extraction failed: %s", exc)
        if not preds and expert_text:
            # Very simple parser:  "broken(cup); on_floor(cup)"
            tokens = [t.strip()
                      for t in expert_text.replace(";", ",").split(",")
                      if t.strip()]
            preds = tokens
            label = expert_text[:32]
        if not preds:
            preds = ["unknown_failure"]
        return preds, label

    def re_synthesize(self) -> Dict[str, Any]:
        """Run MSCA until the Wasserstein delta is below ``ε_spectral``.

        This is the hot-path "autonomic reflex".  For the skeleton we loop
        at most a small number of times so the HTTP call returns quickly;
        in the real experiment this should run in a background task.
        """
        if not self._resync_lock.acquire(blocking=False):
            return {"ok": False, "reason": "already_running"}
        try:
            self.mgr.set_resync_status("running")
            from pyrmdp.synthesis.delta_minimizer import delta_minimize

            history: List[Dict[str, Any]] = []
            MAX_ROUNDS = 5
            for rnd in range(1, MAX_ROUNDS + 1):
                synth = self.mgr.synth
                try:
                    result = delta_minimize(
                        synth.abstract_graph,
                        self.mgr.domain,
                        config=synth.scoring_config,
                        llm_fn=synth.llm_fn,
                    )
                except Exception as exc:
                    logger.warning("delta_minimize failed: %s", exc)
                    break

                n_new = len(result.operators)
                # Incorporate the new operators into the domain
                for op in result.operators:
                    try:
                        self.mgr.domain.actions.append(op.to_action_schema())
                    except AttributeError:
                        # Fallback: synthesized operator object already is an action
                        self.mgr.domain.actions.append(op)
                    self.mgr.dirichlet.setdefault(
                        op.name, DirichletCounts(),
                    )

                # Replace graph with augmented version when available
                if result.augmented_graph is not None:
                    self.mgr.synth.abstract_graph = result.augmented_graph

                delta = self.mgr.refresh_spectral()
                history.append({
                    "round": rnd,
                    "synthesized": n_new,
                    "delta_w": delta,
                    "irreducible": result.is_irreducible,
                })
                logger.info(
                    "[resync r=%d] +%d ops, Δ_W=%.4f, irreducible=%s",
                    rnd, n_new, delta, result.is_irreducible,
                )

                if delta < self.mgr.epsilon_spectral and result.is_irreducible:
                    break
                if n_new == 0:
                    break

            self.mgr.set_resync_status("converged")
            return {"ok": True, "rounds": history,
                    "final_delta_w": self.mgr.spectral_delta}
        finally:
            self._resync_lock.release()
