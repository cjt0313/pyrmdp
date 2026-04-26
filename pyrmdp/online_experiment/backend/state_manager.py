"""
State Manager for the Online Update Experiment
===============================================

Wraps a :class:`pyrmdp.synthesis.iterative_synthesizer.IterativeDomainSynthesizer`
instance so we can mutate the abstract transition graph *online* in response
to execution outcomes reported by a human teleoperator.

The manager is deliberately thread-safe (a single ``RLock``) because the
FastAPI event-loop may handle multiple concurrent requests while a long
MSCA re-synthesis runs in a background task.

Responsibilities
----------------
* Load / initialise the PPDDL ``Domain`` and the abstract graph.
* Keep per-operator Dirichlet pseudo-counts
  ``{action_name: [n_success, n_unchanged, n_failure]}``.
* Maintain the Tabu ledger of severed edges (hallucinated illusions).
* Pick the next action the system wants the human to execute — a thin
  MCTS stand-in that simply cycles through the action list, preferring
  operators with the highest posterior uncertainty (highest LCB width).
* Expose helpers for the updater: ``sever_edge``, ``add_trap_state``,
  ``current_spectrum``, ``wasserstein_delta``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from pyrmdp.synthesis.iterative_synthesizer import (
    IterativeDomainSynthesizer,
    compute_sorted_eigenvalues,
    extract_transition_matrix,
    spectral_distance_wasserstein,
)

try:
    from pyppddl.ppddl.parser import Domain, load_domain
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyPPDDL is required. Install it with "
        "`pip install -e /path/to/POMDPDDL/pyPPDDL`"
    ) from exc

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Dirichlet bookkeeping
# ════════════════════════════════════════════════════════════════════

@dataclass
class DirichletCounts:
    """Posterior pseudo-counts for a single operator.

    Outcome categories (Human 1 reports): success / unchanged / failure.
    We use a symmetric Dirichlet(α=1) prior by default.
    """
    alpha_prior: float = 1.0
    n_success: float = 0.0
    n_unchanged: float = 0.0
    n_failure: float = 0.0

    def total(self) -> float:
        return (
            self.n_success + self.n_unchanged + self.n_failure
            + 3 * self.alpha_prior
        )

    def p_success_mean(self) -> float:
        return (self.n_success + self.alpha_prior) / self.total()

    def p_success_lcb(self, z: float = 1.96) -> float:
        """
        Lower confidence bound on ``P(success)`` using the Beta posterior
        ``Beta(a, b)`` where
            a = n_success + α
            b = n_unchanged + n_failure + 2α
        Approximated by mean − z · sqrt(var).
        """
        a = self.n_success + self.alpha_prior
        b = (self.n_unchanged + self.n_failure) + 2 * self.alpha_prior
        mean = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return max(0.0, mean - z * float(np.sqrt(var)))

    def kinematic_kill(self) -> None:
        """Force ``P_success → 0`` by injecting a large failure mass."""
        self.n_failure += 1e6


# ════════════════════════════════════════════════════════════════════
#  Tabu ledger entry
# ════════════════════════════════════════════════════════════════════

@dataclass
class TabuEntry:
    action_name: str
    edge: Tuple[str, str]
    reason: str
    lcb: float
    timestamp: float = field(default_factory=lambda: __import__("time").time())


# ════════════════════════════════════════════════════════════════════
#  Graph state manager
# ════════════════════════════════════════════════════════════════════

class GraphManager:
    """
    In-memory owner of everything the experiment mutates.

    Parameters
    ----------
    domain_path : str or Path
        Path to the starting PPDDL domain produced by Phase 1.
    epsilon_phys : float
        LCB threshold below which an edge is severed as a hallucinated illusion.
    epsilon_spectral : float
        Wasserstein threshold below which re-synthesis is considered converged.
    output_dir : str or Path
        Where iterative artifacts are saved (shared with the synthesizer).
    """

    def __init__(
        self,
        domain_path: str | Path,
        *,
        epsilon_phys: float = 0.35,
        epsilon_spectral: float = 0.02,
        output_dir: str | Path = "./online_output",
        llm_fn=None,
    ) -> None:
        self._lock = threading.RLock()
        self.epsilon_phys = epsilon_phys
        self.epsilon_spectral = epsilon_spectral
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading PPDDL domain from %s", domain_path)
        self.domain: Domain = load_domain(str(domain_path))

        self.synth = IterativeDomainSynthesizer(
            self.domain,
            epsilon=epsilon_spectral,
            max_iterations=1,      # we re-invoke manually per online step
            output_dir=str(self.output_dir),
            save_intermediates=False,
            llm_fn=llm_fn,
        )
        # Bootstrap the graph once so we have a non-empty spectrum.
        self._bootstrap_graph()

        # Per-operator Dirichlet counts (action_name → counts)
        self.dirichlet: Dict[str, DirichletCounts] = {
            a.name: DirichletCounts() for a in self.domain.actions
        }

        self.tabu: List[TabuEntry] = []
        self.new_predicates: List[str] = []
        self.prev_eigenvalues: np.ndarray = self._current_eigs()
        self.spectral_delta: float = 0.0

        # Action cursor — the "MCTS" suggestion
        self._action_cursor: int = 0
        self._current_action: Optional[str] = self._pick_next_action()

        # Event log (for the baseline/convergence plot)
        self.event_log: List[Dict[str, Any]] = []
        self._resync_status: str = "idle"  # idle | running | converged

    # ──────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _bootstrap_graph(self) -> None:
        """Run Step 2 once to populate ``synth.abstract_graph``."""
        from pyrmdp.core.fodd import FODDManager
        from pyrmdp.synthesis.fodd_builder import (
            build_global_order,
            build_transition_fodd,
            enumerate_abstract_states,
        )
        global_order = build_global_order(self.domain)
        mgr = FODDManager(global_order=global_order)
        _, action_fodds = build_transition_fodd(self.domain.actions, mgr)
        self.synth.abstract_graph = enumerate_abstract_states(
            action_fodds, mgr, self.domain
        )
        logger.info(
            "Bootstrapped abstract graph: %d states, %d edges",
            self.synth.abstract_graph.number_of_nodes(),
            self.synth.abstract_graph.number_of_edges(),
        )

    def _current_eigs(self) -> np.ndarray:
        g = self.synth.abstract_graph
        if g is None or g.number_of_nodes() == 0:
            return np.array([])
        M, _ = extract_transition_matrix(g)
        return compute_sorted_eigenvalues(M)

    # ──────────────────────────────────────────────────────────────
    #  Graph serialization (for the live frontend visualizer)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_policy(action_name: str) -> str:
        """Return ``'human'``, ``'robot1'`` … or ``'unknown'`` from suffix."""
        import re
        if not action_name:
            return "unknown"
        if action_name.startswith("recover"):
            base = "recovery"
        else:
            base = "base"
        m = re.search(r"_(human|robot\d+)$", action_name)
        suffix = m.group(1) if m else "shared"
        return f"{base}:{suffix}"

    @staticmethod
    def _short_state_label(node_id: str, data: Dict[str, Any]) -> str:
        st = data.get("state")
        if data.get("trap"):
            return f"⚠ {data.get('label', node_id)}"
        if st is None:
            return str(node_id)
        true_preds = sorted(getattr(st, "true_predicates", []) or [])
        parts = list(true_preds[:3])
        if len(true_preds) > 3:
            parts.append(f"+{len(true_preds) - 3}")
        return ", ".join(parts) if parts else str(node_id)

    def serialize_graph(self) -> Dict[str, Any]:
        """Dump the abstract transition graph as vis.js-friendly JSON.

        Each node carries the truth-set; each edge carries the action,
        the policy class (``human`` vs ``robotN``, ``base`` vs ``recovery``),
        the latest probability ``prob`` and the running Dirichlet posterior
        statistics for the underlying operator.  Severed (Tabu) operators
        are emitted as dashed edges so they remain visible in the UI.
        """
        with self._lock:
            g = self.synth.abstract_graph
            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []
            if g is None:
                return {"nodes": nodes, "edges": edges}

            tabu_actions = {t.action_name for t in self.tabu}

            for nid, data in g.nodes(data=True):
                trap = bool(data.get("trap"))
                label = self._short_state_label(nid, data)
                st = data.get("state")
                true_preds = sorted(getattr(st, "true_predicates", []) or []) if st else []
                false_preds = sorted(getattr(st, "false_predicates", []) or []) if st else []
                if trap:
                    true_preds = list(data.get("predicates", []))
                title_lines = [f"<b>{nid}</b>"]
                if trap:
                    title_lines.append("<i>(Trap state added online)</i>")
                if true_preds:
                    title_lines.append("<b>True:</b><br>&nbsp;&nbsp;" +
                                       "<br>&nbsp;&nbsp;".join(true_preds))
                if false_preds:
                    title_lines.append("<b>False:</b><br>&nbsp;&nbsp;" +
                                       "<br>&nbsp;&nbsp;".join(false_preds))
                nodes.append({
                    "id": str(nid),
                    "label": label,
                    "title": "<br>".join(title_lines),
                    "trap": trap,
                    "true_predicates": true_preds,
                    "false_predicates": false_preds,
                })

            for u, v, data in g.edges(data=True):
                action = data.get("action", "?")
                policy = self._classify_policy(action)
                prob = float(data.get("prob", 1.0 / 3))
                counts = self.dirichlet.get(action)
                lcb = counts.p_success_lcb() if counts else None
                p_mean = counts.p_success_mean() if counts else None
                severed = action in tabu_actions
                title = (
                    f"<b>{action}</b><br>"
                    f"policy: {policy}<br>"
                    f"prob (graph): {prob:.3f}"
                )
                if counts is not None:
                    title += (
                        f"<br>Dirichlet: s={counts.n_success:.0f} "
                        f"u={counts.n_unchanged:.0f} f={counts.n_failure:.0f}"
                        f"<br>P_success ≈ {p_mean:.3f} "
                        f"(LCB {lcb:.3f}, ε_phys {self.epsilon_phys:.2f})"
                    )
                if severed:
                    title += "<br><b style='color:#f85149'>SEVERED (Tabu)</b>"
                edges.append({
                    "from": str(u),
                    "to": str(v),
                    "action": action,
                    "policy": policy,
                    "prob": prob,
                    "p_success_mean": p_mean,
                    "p_success_lcb": lcb,
                    "severed": severed,
                    "trap": bool(data.get("trap")),
                    "title": title,
                })

            tabu_entries = [
                {
                    "action": t.action_name,
                    "from": t.edge[0],
                    "to": t.edge[1],
                    "reason": t.reason,
                    "lcb": t.lcb,
                    "policy": self._classify_policy(t.action_name),
                }
                for t in self.tabu
            ]

            return {
                "nodes": nodes,
                "edges": edges,
                "tabu": tabu_entries,
                "spectral_distance": self.spectral_delta,
                "epsilon_spectral": self.epsilon_spectral,
                "epsilon_phys": self.epsilon_phys,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
            }

    def _pick_next_action(self) -> Optional[str]:
        """Very small 'planner': cycle through operators, preferring those
        with the widest Dirichlet posterior (least data)."""
        actions = [a.name for a in self.domain.actions
                   if a.name not in {t.action_name for t in self.tabu}]
        if not actions:
            return None
        # Prefer least-sampled (widest CI) action.
        actions.sort(
            key=lambda n: self.dirichlet.get(n, DirichletCounts()).total()
        )
        return actions[0]

    # ──────────────────────────────────────────────────────────────
    #  Read-only status API
    # ──────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            g = self.synth.abstract_graph
            return {
                "current_action": self._current_action,
                "num_nodes": g.number_of_nodes() if g else 0,
                "num_edges": g.number_of_edges() if g else 0,
                "spectral_distance": self.spectral_delta,
                "epsilon_spectral": self.epsilon_spectral,
                "epsilon_phys": self.epsilon_phys,
                "tabu": [
                    {
                        "action": t.action_name,
                        "edge": list(t.edge),
                        "reason": t.reason,
                        "lcb": t.lcb,
                    }
                    for t in self.tabu
                ],
                "new_predicates": list(self.new_predicates),
                "resync_status": self._resync_status,
                "dirichlet": {
                    name: {
                        "n_success": c.n_success,
                        "n_unchanged": c.n_unchanged,
                        "n_failure": c.n_failure,
                        "p_success_mean": c.p_success_mean(),
                        "p_success_lcb": c.p_success_lcb(),
                    }
                    for name, c in self.dirichlet.items()
                },
                "event_log_size": len(self.event_log),
            }

    def event_log_tail(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.event_log[-n:])

    # ──────────────────────────────────────────────────────────────
    #  Mutations
    # ──────────────────────────────────────────────────────────────

    def log_event(self, kind: str, **payload: Any) -> None:
        import time
        with self._lock:
            entry = {"t": time.time(), "kind": kind, **payload}
            self.event_log.append(entry)

    def update_dirichlet(
        self,
        action_name: str,
        *,
        success: float = 0.0,
        unchanged: float = 0.0,
        failure: float = 0.0,
        kinematic_kill: bool = False,
    ) -> DirichletCounts:
        with self._lock:
            counts = self.dirichlet.setdefault(action_name, DirichletCounts())
            counts.n_success += success
            counts.n_unchanged += unchanged
            counts.n_failure += failure
            if kinematic_kill:
                counts.kinematic_kill()
            return counts

    def sever_edges_for(
        self,
        action_name: str,
        *,
        reason: str,
        lcb: float,
    ) -> List[Tuple[str, str]]:
        """Remove all abstract edges contributed by ``action_name`` and
        record them in the Tabu ledger."""
        with self._lock:
            g = self.synth.abstract_graph
            if g is None:
                return []
            removed: List[Tuple[str, str]] = []
            for u, v, data in list(g.edges(data=True)):
                if data.get("action") == action_name:
                    g.remove_edge(u, v)
                    removed.append((u, v))
                    self.tabu.append(
                        TabuEntry(
                            action_name=action_name,
                            edge=(u, v),
                            reason=reason,
                            lcb=lcb,
                        )
                    )
            logger.info(
                "Severed %d edges for %s (reason=%s, lcb=%.3f)",
                len(removed), action_name, reason, lcb,
            )
            return removed

    def add_trap_state(
        self,
        label: str,
        predicates: List[str],
        *,
        from_action: str,
    ) -> str:
        """Attach a new Trap State node to the graph, connected from any
        state where ``from_action`` currently fires."""
        with self._lock:
            g = self.synth.abstract_graph
            node_id = f"trap::{label}::{uuid.uuid4().hex[:6]}"
            g.add_node(node_id, trap=True, label=label, predicates=predicates)
            # Link every precondition-satisfying state into the trap
            attached = 0
            for u, v, data in list(g.edges(data=True)):
                if data.get("action") == from_action:
                    g.add_edge(u, node_id,
                               action=from_action,
                               prob=0.1,
                               trap=True)
                    attached += 1
            self.new_predicates.extend(predicates)
            logger.info(
                "Added trap state %s (attached to %d sources)", node_id, attached,
            )
            return node_id

    # ──────────────────────────────────────────────────────────────
    #  Spectral diagnostics
    # ──────────────────────────────────────────────────────────────

    def refresh_spectral(self) -> float:
        """Recompute eigenvalues and the Wasserstein delta vs. previous."""
        with self._lock:
            curr = self._current_eigs()
            if self.prev_eigenvalues.size == 0 or curr.size == 0:
                self.spectral_delta = 0.0
            else:
                self.spectral_delta = spectral_distance_wasserstein(
                    curr, self.prev_eigenvalues
                )
            self.prev_eigenvalues = curr
            return self.spectral_delta

    def set_resync_status(self, status: str) -> None:
        with self._lock:
            self._resync_status = status

    def advance_action(self) -> Optional[str]:
        with self._lock:
            self._current_action = self._pick_next_action()
            return self._current_action

    @property
    def current_action(self) -> Optional[str]:
        return self._current_action

    # ──────────────────────────────────────────────────────────────
    #  Domain introspection & extension (for expert-guided failures)
    # ──────────────────────────────────────────────────────────────

    def domain_info(self) -> Dict[str, Any]:
        """Types, predicate signatures and existing abstract states."""
        with self._lock:
            types = sorted(self.domain.types.keys()) if self.domain.types else []
            if "object" not in types:
                types = ["object"] + types
            predicates = [
                {
                    "name": p.name,
                    "parameters": [
                        {"name": par.name, "type": par.type}
                        for par in p.parameters
                    ],
                    "signature": (
                        f"({p.name} "
                        + " ".join(
                            f"{par.name} - {par.type}" for par in p.parameters
                        )
                        + ")"
                    ),
                }
                for p in self.domain.predicates
            ]
            pred_names = [p["name"] for p in predicates]
            states: List[Dict[str, Any]] = []
            g = self.synth.abstract_graph
            if g is not None:
                for nid, data in g.nodes(data=True):
                    st = data.get("state")
                    if st is None:
                        continue
                    states.append({
                        "id": str(nid),
                        "true":  sorted(list(getattr(st, "true_predicates", []) or [])),
                        "false": sorted(list(getattr(st, "false_predicates", []) or [])),
                    })
            return {
                "domain": self.domain.name,
                "types": types,
                "predicates": predicates,
                "predicate_names": pred_names,
                "existing_states": states,
                "actions": [a.name for a in self.domain.actions],
            }

    def add_predicate(
        self,
        name: str,
        parameter_types: List[str],
    ) -> Dict[str, Any]:
        """Append a new predicate to the live domain.

        ``parameter_types`` is a list of type names, e.g. ``["object", "location"]``.
        Returns the new predicate signature.  Does NOT re-bootstrap the FODD
        graph on its own — callers should follow with :meth:`rebuild_graph`
        once they are ready for the spectral recomputation.
        """
        from pyppddl.ppddl.parser import Predicate, TypedParam  # type: ignore
        with self._lock:
            name = name.strip()
            if not name:
                raise ValueError("predicate name is required")
            existing = {p.name for p in self.domain.predicates}
            if name in existing:
                raise ValueError(f"predicate '{name}' already exists")
            params = [
                TypedParam(name=f"?x{i}", type=t or "object")
                for i, t in enumerate(parameter_types)
            ]
            pred = Predicate(name=name, parameters=params)
            self.domain.predicates.append(pred)
            logger.info(
                "Added predicate %s(%s)",
                name,
                ", ".join(p.type for p in params),
            )
            self.new_predicates.append(name)
            return {
                "name": name,
                "parameters": [{"name": p.name, "type": p.type} for p in params],
            }

    def add_abstract_state(
        self,
        *,
        true_predicates: List[str],
        false_predicates: List[str],
        from_action: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Create a brand-new abstract state from expert-selected predicates
        and link it into the graph as a trap reachable from ``from_action``."""
        import hashlib
        from pyrmdp.synthesis.fodd_builder import AbstractState
        with self._lock:
            true_fs = frozenset(true_predicates)
            false_fs = frozenset(false_predicates)
            state = AbstractState(true_predicates=true_fs, false_predicates=false_fs)
            key = "::".join([*sorted(true_fs), "|", *sorted(false_fs)])
            nid = f"S_expert_{hashlib.md5(key.encode()).hexdigest()[:6]}"
            g = self.synth.abstract_graph
            if nid in g.nodes:
                return nid  # idempotent
            g.add_node(
                nid,
                state=state,
                trap=True,
                label=label or state.label[:64],
                predicates=sorted(true_fs),
            )
            attached = 0
            if from_action is not None:
                for u, v, data in list(g.edges(data=True)):
                    if data.get("action") == from_action:
                        g.add_edge(u, nid,
                                   action=from_action,
                                   prob=0.1,
                                   trap=True)
                        attached += 1
            logger.info(
                "Added expert abstract state %s (|T|=%d |F|=%d attached=%d)",
                nid, len(true_fs), len(false_fs), attached,
            )
            return nid

    def rebuild_graph(self) -> Dict[str, int]:
        """Re-bootstrap the abstract graph from the current domain.

        Preserves Tabu state (but edges referencing removed predicates may
        reappear).  Use after :meth:`add_predicate` so the new predicate
        participates in enumeration.
        """
        with self._lock:
            self._bootstrap_graph()
            self.refresh_spectral()
            return {
                "num_nodes": self.synth.abstract_graph.number_of_nodes(),
                "num_edges": self.synth.abstract_graph.number_of_edges(),
            }
