"""
pyrmdp.synthesis — Domain Robustification & Symbolic Skill Synthesis

Pipeline:
  Step 0: Domain genesis from RGB + task description (domain_genesis.py)
  Step 1: LLM failure hallucination (llm_failure.py)
  Step 2: Lifted FODD construction from PPDDL (fodd_builder.py)
  Step 3+4: Graph condensation + MSCA bound (graph_analysis.py)
  Step 5: Delta minimization + operator synthesis (delta_minimizer.py)
  Loop:  Iterative Steps 1-5 with spectral convergence (iterative_synthesizer.py)
  Step 6: Multi-policy PPDDL emission (ppddl_emitter.py)
"""

from .llm_config import LLMConfig, load_config, build_llm_fn
from .config import PipelineConfig
from .domain_genesis import generate_initial_domain
from .llm_failure import hallucinate_failures
from .fodd_builder import (
    map_pyppddl_to_pyrmdp,
    build_precondition_fodd,
    build_effect_fodd,
    build_transition_fodd,
    enumerate_abstract_states,
)
from .graph_analysis import (
    condense_to_dag,
    compute_augmentation_bound,
)
from .delta_minimizer import (
    ScoringConfig,
    CandidateEdge,
    delta_minimize,
    calculate_logical_hamming_distance,
)
from .iterative_synthesizer import (
    IterativeDomainSynthesizer,
    IterationRecord,
    extract_transition_matrix,
    compute_sorted_eigenvalues,
    spectral_distance,
    spectral_distance_l2,
    spectral_distance_wasserstein,
)
from .ppddl_emitter import emit_ppddl

__all__ = [
    "LLMConfig",
    "load_config",
    "build_llm_fn",
    "PipelineConfig",
    "generate_initial_domain",
    "hallucinate_failures",
    "map_pyppddl_to_pyrmdp",
    "build_precondition_fodd",
    "build_effect_fodd",
    "build_transition_fodd",
    "enumerate_abstract_states",
    "condense_to_dag",
    "compute_augmentation_bound",
    "ScoringConfig",
    "CandidateEdge",
    "delta_minimize",
    "calculate_logical_hamming_distance",
    "IterativeDomainSynthesizer",
    "IterationRecord",
    "extract_transition_matrix",
    "compute_sorted_eigenvalues",
    "spectral_distance",
    "spectral_distance_l2",
    "spectral_distance_wasserstein",
    "emit_ppddl",
]
