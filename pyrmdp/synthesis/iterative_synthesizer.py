"""
Iterative Domain Robustification Loop (Steps 1–5) with Spectral Convergence

Wraps the per-step modules into a single ``IterativeDomainSynthesizer`` that
repeatedly:
  1. Hallucinate failure modes for newly added operators   (Step 1)
  2. Build the abstract transition graph & extract M_abs   (Step 2)
  3. Condense SCCs into a DAG                              (Step 3)
  4. Compute MSCA bound (sources / sinks)                  (Step 4)
  5. Synthesize recovery operators bridging sink→source     (Step 5)

The loop terminates when the **spectral distance** between consecutive
transition-matrix eigenvalue spectra falls below ε:

    Δ_spectral  =  ‖Λ_current − Λ_prev‖₂  <  ε

Because the matrix dimension can grow across iterations, the shorter
eigenvalue array is zero-padded before the norm is computed.

After the loop converges (or hits ``max_iterations``), Step 6 emits
multi-policy PPDDL.

Usage
-----
>>> from pyrmdp.synthesis.iterative_synthesizer import IterativeDomainSynthesizer
>>> synth = IterativeDomainSynthesizer(domain, epsilon=0.05, max_iterations=10)
>>> ppddl_str = synth.run()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import eigvals

# Internal pipeline modules
from .llm_failure import hallucinate_failures, FailureHallucinationResult
from .fodd_builder import (
    build_global_order,
    build_transition_fodd,
    enumerate_abstract_states,
)
from .graph_analysis import (
    condense_to_dag,
    compute_augmentation_bound,
    CondensationResult,
    AugmentationBound,
)
from .delta_minimizer import (
    delta_minimize,
    ScoringConfig,
    DeltaMinimizationResult,
)
from .ppddl_emitter import emit_ppddl, PolicyExpansionConfig

# pyrmdp core
from ..core.fodd import FODDManager
from ..core.markov import AbstractTransitionMatrix

# pyPPDDL data model
try:
    from pyppddl.ppddl.parser import Domain
except ImportError:
    raise ImportError(
        "pyPPDDL is required. Install it with: pip install -e /path/to/pyPPDDL"
    )

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Spectral-distance utilities
# ════════════════════════════════════════════════════════════════════

def extract_transition_matrix(
    abstract_graph: nx.DiGraph,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the N×N numerical transition matrix from the abstract graph.

    Each edge may carry a ``prob`` attribute; when absent a uniform
    1/3 is assumed (success/unchanged/worse).

    Returns ``(M_abs, state_labels)`` where ``M_abs[i, j]`` is the
    probability of transitioning from state *i* to state *j*.
    Rows are row-normalised to sum to 1.
    """
    nodes = sorted(abstract_graph.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    M = np.zeros((n, n))

    for u, v, data in abstract_graph.edges(data=True):
        M[idx[u], idx[v]] += data.get("prob", 1.0 / 3)

    # Row-normalise → stochastic matrix
    for i in range(n):
        row_sum = M[i].sum()
        if row_sum > 0:
            M[i] /= row_sum
        else:
            M[i, i] = 1.0  # absorbing

    return M, nodes


def compute_sorted_eigenvalues(M: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of *M* and return them sorted by descending
    absolute magnitude.
    """
    eigs = eigvals(M)
    return np.array(sorted(np.abs(eigs), reverse=True))


def spectral_distance(
    current: np.ndarray,
    previous: np.ndarray,
) -> float:
    """
    L₂ norm of the difference between two eigenvalue spectra.

    The shorter array is **zero-padded** at the tail so that both
    arrays have the same length before subtracting.
    """
    max_len = max(len(current), len(previous))
    padded_curr = np.zeros(max_len)
    padded_prev = np.zeros(max_len)
    padded_curr[: len(current)] = current
    padded_prev[: len(previous)] = previous
    return float(np.linalg.norm(padded_curr - padded_prev))


# ════════════════════════════════════════════════════════════════════
#  Per-iteration record
# ════════════════════════════════════════════════════════════════════

@dataclass
class IterationRecord:
    """Diagnostics for one pass through Steps 1–5."""
    iteration: int
    num_states: int = 0
    num_edges: int = 0
    num_sccs: int = 0
    is_strongly_connected: bool = False
    num_sources: int = 0
    num_sinks: int = 0
    msca_bound: int = 0
    spectral_distance: Optional[float] = None
    eigenvalues: Optional[List[float]] = None
    step1_actions_processed: int = 0
    step5_operators_synthesized: int = 0
    step5_is_irreducible: bool = False
    elapsed: float = 0.0


# ════════════════════════════════════════════════════════════════════
#  Iterative Domain Synthesizer
# ════════════════════════════════════════════════════════════════════

class IterativeDomainSynthesizer:
    """
    Manages the iterative robustification loop (Steps 1 → 5) with a
    **spectral-distance** stopping criterion, then emits multi-policy
    PPDDL in Step 6.

    Parameters
    ----------
    domain : Domain
        The pyPPDDL Domain object (from Step 0 or loaded from file).
        **Mutated in place** as new predicates/types/actions are added.
    epsilon : float
        Convergence threshold for spectral distance (default 0.05).
    max_iterations : int
        Hard cap on loop iterations (default 10).
    failure_prob : float
        Probability assigned to failure branches in Step 1 (default 0.1).
    scoring_config : ScoringConfig, optional
        Delta-minimization scoring parameters for Step 5.
    emission_config : PolicyExpansionConfig, optional
        Multi-policy expansion parameters for Step 6.
    llm_fn : callable, optional
        ``fn(prompt: str) → str``. If *None*, built from ``llm.yaml``
        / env-vars at first use.
    output_dir : str or Path, optional
        Directory for final PPDDL + optional intermediates.
    save_intermediates : bool
        Whether to persist per-step JSON/GraphML alongside the PPDDL.

    Attributes
    ----------
    history : list[IterationRecord]
        Per-iteration diagnostics.
    converged : bool
        ``True`` if the loop terminated via spectral convergence.
    """

    def __init__(
        self,
        domain: Domain,
        *,
        epsilon: float = 0.05,
        max_iterations: int = 10,
        failure_prob: float = 0.1,
        scoring_config: Optional[ScoringConfig] = None,
        emission_config: Optional[PolicyExpansionConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        output_dir: str = "./pipeline_output",
        save_intermediates: bool = False,
    ) -> None:
        self.domain = domain
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.failure_prob = failure_prob
        self.scoring_config = scoring_config or ScoringConfig()
        self.emission_config = emission_config or PolicyExpansionConfig()
        self.llm_fn = llm_fn
        self.output_dir = Path(output_dir)
        self.save_intermediates = save_intermediates

        self._prev_eigenvalues: np.ndarray = np.array([])
        self.history: List[IterationRecord] = []
        self.converged: bool = False

        # Filled after run()
        self.abstract_graph: Optional[nx.DiGraph] = None
        self.condensation: Optional[CondensationResult] = None
        self.augmentation_bound: Optional[AugmentationBound] = None
        self.delta_result: Optional[DeltaMinimizationResult] = None
        self.ppddl_output: Optional[str] = None

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

    def _ensure_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _save_json(self, filename: str, data: Any) -> None:
        if not self.save_intermediates:
            return
        out = self._ensure_dir()
        path = out / filename

        def _default(obj):
            if isinstance(obj, (set, frozenset)):
                return sorted(obj)
            return str(obj)

        path.write_text(
            json.dumps(data, indent=2, default=_default), encoding="utf-8"
        )
        logger.info(f"  ✓ Saved: {path}")

    def _save_graph(self, filename: str, graph: nx.DiGraph) -> None:
        if not self.save_intermediates:
            return
        out = self._ensure_dir()
        path = out / filename
        G = graph.copy()
        for _, _, d in G.edges(data=True):
            for k, v in list(d.items()):
                if isinstance(v, (set, frozenset)):
                    d[k] = ",".join(sorted(str(x) for x in v))
                elif not isinstance(v, (str, int, float, bool)):
                    d[k] = str(v)
        for _, d in G.nodes(data=True):
            for k, v in list(d.items()):
                if not isinstance(v, (str, int, float, bool)):
                    d[k] = str(v)
        nx.write_graphml(G, str(path))
        logger.info(f"  ✓ Saved: {path}")

    def _save_text(self, filename: str, text: str) -> None:
        if not self.save_intermediates:
            return
        out = self._ensure_dir()
        path = out / filename
        path.write_text(text, encoding="utf-8")
        logger.info(f"  ✓ Saved: {path}")

    # ──────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────

    def run(self) -> str:
        """
        Execute the iterative robustification loop (Steps 1–5),
        then emit multi-policy PPDDL (Step 6).

        Returns
        -------
        str
            The final PPDDL domain string.
        """
        logger.info("═" * 60)
        logger.info("Iterative Domain Robustification Loop  (Steps 1 → 5)")
        logger.info(
            f"  ε = {self.epsilon},  max_iter = {self.max_iterations},  "
            f"failure_prob = {self.failure_prob}"
        )
        logger.info(
            f"  Domain: {self.domain.name}, "
            f"{len(self.domain.types)} types, "
            f"{len(self.domain.predicates)} predicates, "
            f"{len(self.domain.actions)} actions"
        )
        logger.info("═" * 60)

        # Track the set of action names present *before* this iteration
        # so that Step 1 only processes newly added operators.
        known_action_names: set = set()
        timings: Dict[str, float] = {}

        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            rec = IterationRecord(iteration=iteration)

            logger.info("")
            logger.info(f"──── Iteration {iteration} ────")

            # ── Step 1: Failure Hallucination (new operators only) ──
            new_actions = [
                a for a in self.domain.actions
                if a.name not in known_action_names
            ]
            known_action_names.update(a.name for a in self.domain.actions)

            if new_actions:
                logger.info(
                    f"[Step 1] Failure Hallucination  "
                    f"({len(new_actions)} new actions)"
                )
                t0 = time.time()
                self.domain, failure_results = hallucinate_failures(
                    self.domain,
                    llm_fn=self.llm_fn,
                    failure_prob=self.failure_prob,
                )
                timings[f"iter{iteration}_step1"] = time.time() - t0

                rec.step1_actions_processed = len(failure_results)
                logger.info(
                    f"  Hallucinated failures for "
                    f"{len(failure_results)} actions"
                )
                for fr in failure_results:
                    logger.info(
                        f"    {fr.action_name}: "
                        f"+{len(fr.new_predicates)} preds, "
                        f"+{len(fr.new_types)} types"
                    )

                self._save_json(
                    f"iter{iteration}_step1_failures.json",
                    [
                        {
                            "action": fr.action_name,
                            "new_predicates": [
                                p.name for p in fr.new_predicates
                            ],
                            "new_types": fr.new_types,
                            "raw_response": fr.raw_llm_response[:500],
                        }
                        for fr in failure_results
                    ],
                )
            else:
                logger.info("[Step 1] No new actions — skipping hallucination")

            # ── Step 2: Abstract Graph → Transition Matrix ──
            logger.info("[Step 2] Abstract Graph & Matrix Extraction")
            t0 = time.time()
            global_order = build_global_order(self.domain)
            manager = FODDManager(global_order=global_order)
            composed_root, action_fodds = build_transition_fodd(
                self.domain.actions, manager
            )
            abstract_graph = enumerate_abstract_states(
                action_fodds, manager, self.domain
            )
            self.abstract_graph = abstract_graph
            timings[f"iter{iteration}_step2"] = time.time() - t0

            n_states = abstract_graph.number_of_nodes()
            n_edges = abstract_graph.number_of_edges()
            rec.num_states = n_states
            rec.num_edges = n_edges
            logger.info(
                f"  FODD nodes: {len(manager.nodes)}, "
                f"States: {n_states}, Edges: {n_edges}"
            )

            self._save_graph(
                f"iter{iteration}_step2_abstract_graph.graphml",
                abstract_graph,
            )
            # Save abstract state details
            states_info = []
            for nid, data in abstract_graph.nodes(data=True):
                st = data.get("state")
                if st:
                    states_info.append({
                        "id": nid,
                        "true": sorted(st.true_predicates),
                        "false": sorted(st.false_predicates),
                        "label": st.label,
                    })
            self._save_json(
                f"iter{iteration}_step2_abstract_states.json", states_info,
            )

            # ── Extract transition matrix & eigenvalues ──
            M_abs, state_labels = extract_transition_matrix(abstract_graph)
            eigenvalues = compute_sorted_eigenvalues(M_abs)
            rec.eigenvalues = eigenvalues.tolist()

            # ── Stopping Criterion: Spectral Distance ──
            if self._prev_eigenvalues.size > 0:
                delta_spectral = spectral_distance(
                    eigenvalues, self._prev_eigenvalues,
                )
                rec.spectral_distance = delta_spectral
                logger.info(
                    f"  Spectral distance: Δ = {delta_spectral:.6f}  "
                    f"(ε = {self.epsilon})"
                )

                if delta_spectral < self.epsilon:
                    logger.info("  ✓ Spectral Convergence Reached.")
                    rec.elapsed = time.time() - iter_start
                    self.converged = True
                    self.history.append(rec)
                    break
            else:
                logger.info("  (first iteration — no previous spectrum)")

            self._prev_eigenvalues = eigenvalues

            # ── Step 3: SCC Condensation ──
            logger.info("[Step 3] SCC Condensation")
            t0 = time.time()
            condensation = condense_to_dag(abstract_graph)
            self.condensation = condensation
            timings[f"iter{iteration}_step3"] = time.time() - t0

            rec.num_sccs = condensation.num_sccs
            rec.is_strongly_connected = condensation.is_strongly_connected
            logger.info(
                f"  SCCs: {condensation.num_sccs}, "
                f"Strongly connected: {condensation.is_strongly_connected}"
            )

            self._save_json(
                f"iter{iteration}_step3_condensation.json",
                {
                    "num_sccs": condensation.num_sccs,
                    "is_strongly_connected": condensation.is_strongly_connected,
                    "sccs": {
                        str(k): v
                        for k, v in condensation.scc_state_map.items()
                    },
                    "labels": condensation.scc_label_map,
                },
            )
            self._save_graph(
                f"iter{iteration}_step3_dag.graphml", condensation.dag,
            )

            # ── Step 4: MSCA Bound ──
            logger.info("[Step 4] MSCA Bound")
            t0 = time.time()
            aug_bound = compute_augmentation_bound(condensation)
            self.augmentation_bound = aug_bound
            timings[f"iter{iteration}_step4"] = time.time() - t0

            rec.num_sources = len(aug_bound.sources)
            rec.num_sinks = len(aug_bound.sinks)
            rec.msca_bound = aug_bound.bound
            logger.info(
                f"  Sources: {len(aug_bound.sources)}, "
                f"Sinks: {len(aug_bound.sinks)}, "
                f"MSCA bound: {aug_bound.bound}"
            )

            self._save_json(
                f"iter{iteration}_step4_augmentation_bound.json",
                {
                    "sources": aug_bound.sources,
                    "sinks": aug_bound.sinks,
                    "bound": aug_bound.bound,
                    "is_already_irreducible": aug_bound.is_already_irreducible,
                },
            )

            if aug_bound.is_already_irreducible:
                logger.info("  Graph is already irreducible!")
                rec.step5_is_irreducible = True
                rec.elapsed = time.time() - iter_start
                self.converged = True
                self.history.append(rec)
                break

            # ── Step 5: Delta Minimization & Synthesis ──
            logger.info("[Step 5] Delta Minimization & Synthesis")
            t0 = time.time()
            delta_result = delta_minimize(
                abstract_graph,
                self.domain,
                llm_fn=self.llm_fn,
                config=self.scoring_config,
            )
            self.delta_result = delta_result
            timings[f"iter{iteration}_step5"] = time.time() - t0

            rec.step5_operators_synthesized = len(delta_result.operators)
            rec.step5_is_irreducible = delta_result.is_irreducible
            logger.info(
                f"  Synthesized: {len(delta_result.operators)} "
                f"recovery operators, "
                f"Irreducible: {delta_result.is_irreducible}"
            )

            self._save_json(
                f"iter{iteration}_step5_synthesized_operators.json",
                [
                    {
                        "name": op.name,
                        "sink_scc": op.sink_scc,
                        "source_scc": op.source_scc,
                        "delta": op.delta,
                        "nominal_add": [list(t) for t in op.nominal_add],
                        "nominal_del": [list(t) for t in op.nominal_del],
                        "failure_add": [list(t) for t in op.failure_add],
                        "failure_del": [list(t) for t in op.failure_del],
                    }
                    for op in delta_result.operators
                ],
            )
            self._save_json(
                f"iter{iteration}_step5_stats.json", delta_result.stats,
            )

            # If already irreducible after Step 5, we can stop early
            if delta_result.is_irreducible:
                logger.info("  Graph is now irreducible after Step 5.")
                rec.elapsed = time.time() - iter_start
                self.converged = True
                self.history.append(rec)
                break

            rec.elapsed = time.time() - iter_start
            self.history.append(rec)

        else:
            # Loop exhausted without convergence
            logger.warning(
                f"Reached max_iterations ({self.max_iterations}) "
                f"without spectral convergence."
            )

        # ── Post-Loop: Step 6 — Multi-Policy PPDDL Emission ──
        logger.info("")
        logger.info("═" * 60)
        logger.info(
            f"Step 6: Multi-Policy PPDDL Emission "
            f"(K={self.emission_config.num_robot_policies})"
        )
        logger.info("═" * 60)

        out = self._ensure_dir()
        output_path = str(out / "robustified.ppddl")

        self.ppddl_output = emit_ppddl(
            self.domain,
            output_path=output_path,
            config=self.emission_config,
        )

        # ── Save pipeline summary ──
        total_time = sum(timings.values())
        logger.info("")
        logger.info("═" * 60)
        logger.info("Pipeline Complete (iterative loop)")
        logger.info("═" * 60)
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Converged: {self.converged}")
        logger.info(f"  Iterations: {len(self.history)}")
        logger.info(f"  Total actions: {len(self.domain.actions)}")
        logger.info(f"  Total time: {total_time:.1f}s")

        summary_data = self.summary()
        summary_data["timings"] = timings
        summary_data["output_path"] = output_path
        self._save_json("pipeline_summary.json", summary_data)

        return self.ppddl_output

    # ──────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a JSON-serialisable summary of the run."""
        return {
            "converged": self.converged,
            "iterations": len(self.history),
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "total_actions": len(self.domain.actions),
            "spectral_distances": [
                r.spectral_distance
                for r in self.history
                if r.spectral_distance is not None
            ],
            "msca_bounds": [
                r.msca_bound
                for r in self.history
                if r.msca_bound > 0
            ],
            "per_iteration": [
                {
                    "iteration": r.iteration,
                    "states": r.num_states,
                    "edges": r.num_edges,
                    "sccs": r.num_sccs,
                    "sources": r.num_sources,
                    "sinks": r.num_sinks,
                    "msca_bound": r.msca_bound,
                    "spectral_distance": r.spectral_distance,
                    "step1_actions": r.step1_actions_processed,
                    "step5_operators": r.step5_operators_synthesized,
                    "elapsed_s": round(r.elapsed, 2),
                }
                for r in self.history
            ],
        }
