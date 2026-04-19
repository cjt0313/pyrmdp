"""
Iterative Domain Robustification Loop (Steps 1–5) with Spectral Convergence

Wraps the per-step modules into a single ``IterativeDomainSynthesizer`` that
repeatedly:
  1. Hallucinate failure modes for newly added operators   (Step 1)
  2. Build the abstract transition graph & extract M_abs   (Step 2)
  3. Condense SCCs into a DAG                              (Step 3)
  4. Compute MSCA bound (sources / sinks)                  (Step 4)
  5. Synthesize recovery operators bridging sink→source     (Step 5)

The loop terminates when the **Wasserstein spectral distance** between
consecutive transition-matrix eigenvalue spectra falls below ε:

    Δ_W  =  W₁(Λ_current, Λ_prev)  <  ε

Wasserstein (Earth Mover's) distance is **dimension-invariant**: it
treats sorted eigenvalue arrays as empirical distributions and computes
the optimal transport cost, with no zero-padding needed.  It provides
the tightest convergence signal — typically 1–2 orders of magnitude
smaller than cosine distance.

The ``max_recovery_per_iter`` budget cap (set via ``ScoringConfig``)
limits how many recovery operators Step 5 can emit per outer-loop
iteration.  With budget = 1, the loop adds one operator at a time,
spreading the repair across many iterations and producing the
multi-iteration spectral-convergence curves needed for the paper.

After the loop converges (or hits ``max_iterations``), Step 6 emits
multi-policy PPDDL.

Usage
-----
>>> from pyrmdp.synthesis.iterative_synthesizer import IterativeDomainSynthesizer
>>> synth = IterativeDomainSynthesizer(domain, epsilon=0.02, max_iterations=10)
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
from scipy.stats import wasserstein_distance as _wasserstein_1d

from .llm_config import get_global_tracker, reset_global_tracker

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


def _zero_pad(a: np.ndarray, b: np.ndarray):
    """Return zero-padded copies of *a* and *b* with equal length."""
    max_len = max(len(a), len(b))
    pa, pb = np.zeros(max_len), np.zeros(max_len)
    pa[: len(a)] = a
    pb[: len(b)] = b
    return pa, pb


def spectral_distance(
    current: np.ndarray,
    previous: np.ndarray,
) -> float:
    """
    Cosine distance between two eigenvalue spectra.

    Returns ``1 − cos(Λ_curr, Λ_prev)`` after zero-padding the shorter
    array.  Cosine distance is **dimension-invariant**: proportional
    growth of the state-space (which adds eigenvalue-1 entries for new
    absorbing states) does not inflate the metric the way raw L₂ does.

    Range: [0, 2].  0 = identical spectral shape.
    """
    pa, pb = _zero_pad(current, previous)
    na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
    if na == 0 or nb == 0:
        return 0.0 if (na == 0 and nb == 0) else 1.0
    cos_sim = float(np.clip(np.dot(pa, pb) / (na * nb), -1.0, 1.0))
    return 1.0 - cos_sim


def spectral_distance_l2(
    current: np.ndarray,
    previous: np.ndarray,
) -> float:
    """
    Raw L₂ norm between two zero-padded eigenvalue spectra.

    Kept for backward compatibility and logging; **not** recommended
    as a convergence criterion because it is dominated by state-space
    dimension growth rather than structural change.
    """
    pa, pb = _zero_pad(current, previous)
    return float(np.linalg.norm(pa - pb))


def spectral_distance_wasserstein(
    current: np.ndarray,
    previous: np.ndarray,
) -> float:
    """
    1-D Wasserstein (Earth Mover's) distance between two eigenvalue spectra.

    Uses ``scipy.stats.wasserstein_distance`` which treats each sorted
    eigenvalue array as an empirical distribution and computes the optimal
    transport cost.  **No zero-padding** is needed — the function handles
    arrays of different lengths natively by treating them as discrete
    distributions with uniform weights.

    This metric captures the *transport cost* of reshaping one spectrum
    into the other and is robust to dimension growth.
    """
    return float(_wasserstein_1d(current, previous))


# ════════════════════════════════════════════════════════════════════
#  Operator NL Translation
# ════════════════════════════════════════════════════════════════════

_NL_SYSTEM = (
    "You are a concise robotics-domain translator. "
    "Given a PDDL recovery operator's preconditions and effects, "
    "produce ONE short English sentence (max 20 words) that describes "
    "what the robot does. Be precise and clear. "
    "Reply with ONLY the sentence, no quotes, no explanation."
)


def _translate_operators_to_nl(
    operators: list,
    llm_fn,
) -> Dict[str, str]:
    """Batch-translate operators to natural language.

    Parameters
    ----------
    operators : list[dict]
        Serialised operator dicts with keys ``name``, ``precondition_atoms``,
        ``nominal_add``, ``nominal_del``, ``parameters``.
    llm_fn : callable
        ``fn(prompt, *, system=...) → str``

    Returns
    -------
    dict mapping operator name → NL sentence
    """
    results: Dict[str, str] = {}
    if not operators or llm_fn is None:
        return results

    # Build one batch prompt for all operators
    lines = []
    for i, op in enumerate(operators):
        preconds = op.get("precondition_atoms", [])
        adds = op.get("nominal_add", [])
        dels = op.get("nominal_del", [])
        params = op.get("parameters", [])
        param_str = ", ".join(f"{p['name']}: {p['type']}" for p in params)
        pre_str = ", ".join(str(a) for a in preconds) or "(none)"
        add_str = ", ".join(str(a) for a in adds) or "(none)"
        del_str = ", ".join(str(a) for a in dels) or "(none)"
        phase_tag = op.get("phase_tag", "")
        lines.append(
            f"[{i}] {op['name']} (params: {param_str})\n"
            f"  Source: {phase_tag or 'hamming'}\n"
            f"  Preconditions: {pre_str}\n"
            f"  Add effects: {add_str}\n"
            f"  Del effects: {del_str}"
        )

    prompt = (
        "Translate each operator below into ONE short English sentence.\n"
        "Format: one line per operator, starting with the index number.\n"
        "Example: [0] Pick up the cup from the table.\n\n"
        + "\n\n".join(lines)
    )

    try:
        raw = llm_fn(prompt, system=_NL_SYSTEM)
        # Parse: each line starts with [i]
        import re
        for m in re.finditer(r'\[(\d+)\]\s*(.+)', raw):
            idx = int(m.group(1))
            sentence = m.group(2).strip().rstrip(".")  + "."
            if idx < len(operators):
                results[operators[idx]["name"]] = sentence
    except Exception as exc:
        logger.warning(f"NL operator translation failed: {exc}")

    return results


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
    spectral_distance_l2: Optional[float] = None
    spectral_distance_wasserstein: Optional[float] = None
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
        Convergence threshold for Wasserstein spectral distance (default 0.02).
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
        epsilon: float = 0.02,
        max_iterations: int = 10,
        failure_prob: float = 0.1,
        scoring_config: Optional[ScoringConfig] = None,
        emission_config: Optional[PolicyExpansionConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        output_dir: str = "./pipeline_output",
        save_intermediates: bool = False,
        enable_mutex_pruning: bool = False,
        mutex_groups: Optional[List] = None,
        mutex_pairwise_rules: Optional[List] = None,
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
        self.enable_mutex_pruning = enable_mutex_pruning

        # ── Mutex groups (SAS+ exactly-one) ──
        self._mutex_groups: List = list(mutex_groups or [])
        self._mutex_pairwise_rules: List = list(mutex_pairwise_rules or [])
        # Track predicate names known when groups were last queried
        self._mutex_pred_snapshot: Set[str] = {
            p.name for p in domain.predicates
        }

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
        # Sanitize edge attributes for GraphML (only str/int/float/bool)
        for _, _, d in G.edges(data=True):
            for k, v in list(d.items()):
                if isinstance(v, (set, frozenset)):
                    d[k] = ",".join(sorted(str(x) for x in v))
                elif not isinstance(v, (str, int, float, bool)):
                    d[k] = str(v)
        # Sanitize node attributes
        for _, d in G.nodes(data=True):
            for k, v in list(d.items()):
                if isinstance(v, (set, frozenset)):
                    d[k] = ",".join(sorted(str(x) for x in v))
                elif not isinstance(v, (str, int, float, bool)):
                    d[k] = str(v)
        # Sanitize graph-level attributes (e.g. nx.condensation adds 'mapping')
        for k, v in list(G.graph.items()):
            if isinstance(v, (set, frozenset)):
                G.graph[k] = ",".join(sorted(str(x) for x in v))
            elif isinstance(v, (dict, list)):
                del G.graph[k]  # drop complex graph attrs unsupported by GraphML
            elif not isinstance(v, (str, int, float, bool)):
                G.graph[k] = str(v)
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
        # ── Clean stale iter* artefacts from previous runs ──
        out = self._ensure_dir()
        reset_global_tracker()  # Fresh usage tracking for this run
        for stale in out.glob("iter*"):
            stale.unlink(missing_ok=True)
        for stale in out.glob("robustified.ppddl"):
            stale.unlink(missing_ok=True)
        for stale in out.glob("pipeline_summary.json"):
            stale.unlink(missing_ok=True)

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
            # Frontier-Only Policy: each operator (base or recovery) is
            # hallucinated exactly once, in the iteration it first appears.
            # `known_action_names` prevents re-processing.
            new_actions = [
                a for a in self.domain.actions
                if a.name not in known_action_names
            ]
            known_action_names.update(a.name for a in self.domain.actions)

            if new_actions:
                n_base = sum(1 for a in new_actions if not a.name.startswith("recover_"))
                n_recov = len(new_actions) - n_base
                logger.info(
                    f"[Step 1] Failure Hallucination  "
                    f"({len(new_actions)} new actions: "
                    f"{n_base} base, {n_recov} recovery)"
                )
                t0 = time.time()
                self.domain, failure_results = hallucinate_failures(
                    self.domain,
                    llm_fn=self.llm_fn,
                    failure_prob=self.failure_prob,
                    action_names={a.name for a in new_actions},
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

                # ── Re-query mutex groups if new predicates appeared ──
                if self._mutex_groups:
                    current_preds = {p.name for p in self.domain.predicates}
                    new_preds = current_preds - self._mutex_pred_snapshot
                    if new_preds:
                        logger.info(
                            "  New predicates detected (%s) — "
                            "re-querying mutex groups …",
                            ", ".join(sorted(new_preds)),
                        )
                        try:
                            from ..pruning.llm_axiom import generate_mutex_groups
                            pred_sigs = []
                            for pred in self.domain.predicates:
                                sig_parts = [pred.name] + [
                                    p.name for p in pred.parameters
                                ]
                                pred_sigs.append(" ".join(sig_parts))
                            groups, pw_rules = generate_mutex_groups(
                                pred_sigs,
                                llm_fn=self.llm_fn,
                                valid_predicate_names=current_preds,
                            )
                            self._mutex_groups = groups
                            self._mutex_pairwise_rules = pw_rules
                            self._mutex_pred_snapshot = current_preds
                            logger.info(
                                "  Updated: %d mutex groups, %d pairwise rules",
                                len(groups), len(pw_rules),
                            )
                        except Exception as exc:
                            logger.warning(
                                "  Mutex group re-query failed: %s", exc
                            )

                # ── Patch operator effects with mutex groups ──
                if self._mutex_groups:
                    from ..pruning.llm_axiom import patch_operator_effects
                    patched_count = 0
                    for i, action in enumerate(self.domain.actions):
                        patched = patch_operator_effects(
                            action, self._mutex_groups, self.domain,
                        )
                        if patched is not action:
                            self.domain.actions[i] = patched
                            patched_count += 1
                    if patched_count:
                        logger.info(
                            "  Patched %d actions with mutex group deletes",
                            patched_count,
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

            # ── R5: LLM-based mutex pruning (optional) ──
            if self.enable_mutex_pruning:
                from ..pruning.llm_axiom import (
                    generate_mutex_rules,
                    prune_with_mutexes,
                    rules_to_dict,
                )
                pred_names = [p.name for p in self.domain.predicates]
                mutex_rules = generate_mutex_rules(
                    pred_names, llm_fn=self.llm_fn,
                )
                mutex_result = prune_with_mutexes(abstract_graph, mutex_rules)
                logger.info(
                    f"  R5 mutex pruning: {mutex_result.original_count} → "
                    f"{abstract_graph.number_of_nodes()} states "
                    f"(-{mutex_result.pruned_count} pruned by "
                    f"{len(mutex_rules)} rules)"
                )
                self._save_json(
                    f"iter{iteration}_step2_mutex_rules.json",
                    {
                        "rules": rules_to_dict(mutex_rules),
                        "pruned_states": mutex_result.pruned_states,
                        "original_count": mutex_result.original_count,
                        "pruned_count": mutex_result.pruned_count,
                    },
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

            # ── Stopping Criterion: Spectral Distance (Wasserstein) ──
            if self._prev_eigenvalues.size > 0:
                delta_cos = spectral_distance(
                    eigenvalues, self._prev_eigenvalues,
                )
                delta_l2 = spectral_distance_l2(
                    eigenvalues, self._prev_eigenvalues,
                )
                delta_w = spectral_distance_wasserstein(
                    eigenvalues, self._prev_eigenvalues,
                )
                rec.spectral_distance = delta_cos
                rec.spectral_distance_l2 = delta_l2
                rec.spectral_distance_wasserstein = delta_w
                logger.info(
                    f"  Spectral distance: Δ_W = {delta_w:.6f}  "
                    f"(ε = {self.epsilon}), Δ_cos = {delta_cos:.6f}, "
                    f"Δ_L2 = {delta_l2:.4f}"
                )

                if delta_w < self.epsilon:
                    logger.info("  ✓ Spectral Convergence Reached (Wasserstein).")
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
                logger.info("  Graph is already irreducible — skipping Step 5")
                rec.step5_is_irreducible = True
                rec.elapsed = time.time() - iter_start
                self.history.append(rec)
                continue

            # ── Step 5: Delta Minimization & Synthesis ──
            logger.info("[Step 5] Delta Minimization & Synthesis")
            t0 = time.time()
            delta_result = delta_minimize(
                abstract_graph,
                self.domain,
                config=self.scoring_config,
                mutex_groups=self._mutex_groups or None,
                llm_fn=self.llm_fn,
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

            op_dicts = [
                {
                    "name": op.name,
                    "sink_scc": op.sink_scc,
                    "source_scc": op.source_scc,
                    "sink_node": op.sink_node,
                    "source_node": op.source_node,
                    "phase_tag": op.phase_tag,
                    "delta": op.delta,
                    "parameters": [
                        {"name": p.name, "type": p.type}
                        for p in op.parameters
                    ],
                    "precondition_atoms": [list(t) for t in op.precondition_atoms],
                    "nominal_add": [list(t) for t in op.nominal_add],
                    "nominal_del": [list(t) for t in op.nominal_del],
                }
                for op in delta_result.operators
            ]

            # ── Step 5b: Translate operators to natural language ──
            query_fn = self.llm_fn
            if query_fn is None:
                try:
                    from .llm_config import build_llm_fn
                    query_fn = build_llm_fn()
                except Exception:
                    query_fn = None
            nl_map = _translate_operators_to_nl(op_dicts, query_fn)
            for od in op_dicts:
                od["nl_description"] = nl_map.get(od["name"], "")

            self._save_json(
                f"iter{iteration}_step5_synthesized_operators.json",
                op_dicts,
            )
            self._save_json(
                f"iter{iteration}_step5_stats.json", delta_result.stats,
            )

            # Save the augmented graph for visualization/verification
            if delta_result.augmented_graph is not None:
                aug_path = self._ensure_dir() / f"iter{iteration}_step5_augmented_graph.graphml"
                G = delta_result.augmented_graph.copy()
                # Sanitize ALL node/edge attributes for GraphML
                # (AbstractState, sets, dicts, etc. must become strings)
                _basic = (str, int, float, bool)
                for n in G.nodes():
                    for k, v in list(G.nodes[n].items()):
                        if not isinstance(v, _basic):
                            G.nodes[n][k] = str(v)
                for u, v, d in G.edges(data=True):
                    for k, val in list(d.items()):
                        if not isinstance(val, _basic):
                            d[k] = str(val)
                nx.write_graphml(G, str(aug_path))
                logger.info(f"  Saved augmented graph to {aug_path.name}")

            if delta_result.is_irreducible:
                logger.info("  Graph is now irreducible after Step 5.")

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

        # ── Save LLM usage report ──
        tracker = get_global_tracker()
        tracker.save(out / "llm_usage.json")

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
            "spectral_metric": "wasserstein",
            "spectral_distances": [
                r.spectral_distance
                for r in self.history
                if r.spectral_distance is not None
            ],
            "spectral_distances_l2": [
                r.spectral_distance_l2
                for r in self.history
                if r.spectral_distance_l2 is not None
            ],
            "spectral_distances_wasserstein": [
                r.spectral_distance_wasserstein
                for r in self.history
                if r.spectral_distance_wasserstein is not None
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
                    "is_strongly_connected": r.is_strongly_connected,
                    "sources": r.num_sources,
                    "sinks": r.num_sinks,
                    "msca_bound": r.msca_bound,
                    "spectral_distance": r.spectral_distance,
                    "spectral_distance_l2": r.spectral_distance_l2,
                    "spectral_distance_wasserstein": r.spectral_distance_wasserstein,
                    "eigenvalues": r.eigenvalues,
                    "step1_actions": r.step1_actions_processed,
                    "step5_operators": r.step5_operators_synthesized,
                    "step5_is_irreducible": r.step5_is_irreducible,
                    "elapsed_s": round(r.elapsed, 2),
                }
                for r in self.history
            ],
        }
