# pyrmdp — Lifted FODDs + Domain Robustification Pipeline

`pyrmdp` is a Python library for **Relational Markov Decision Processes (RMDPs)** built on **First-Order Decision Diagrams (FODDs)**. It works directly with the **lifted** (relational) representation — no exponential grounding required.

On top of the FODD core, pyrmdp ships a **synthesis pipeline** that takes a deterministic PDDL domain, hallucinate failure modes via an LLM, extracts an abstract Markov chain, and synthesizes the minimum set of recovery operators to guarantee strong connectivity — emitting a multi-policy PPDDL with reward-annotated human/robot branches.

## Table of Contents

- [1. Core: FODDs & Reductions](#1-core-fodds--reductions)
  - [1.1 Relational MDPs](#11-relational-mdps)
  - [1.2 FODD Reduction Rules (R1–R5)](#12-fodd-reduction-rules-r1r5)
- [2. Synthesis Pipeline](#2-synthesis-pipeline)
- [3. Architecture](#3-architecture)
  - [3.1 Project Structure](#31-project-structure)
  - [3.2 Core Modules](#32-core-modules)
  - [3.3 Synthesis Modules](#33-synthesis-modules)
- [4. Installation](#4-installation)
- [5. Quick Start](#5-quick-start)
  - [5.1 FODD Visualization](#51-fodd-visualization)
  - [5.2 Running the Synthesis Pipeline](#52-running-the-synthesis-pipeline)
- [6. References](#6-references)

---

## 1. Core: FODDs & Reductions

### 1.1 Relational MDPs

| | Grounded MDP | Relational MDP |
|---|---|---|
| **States** | Flat identifiers (s1, s2, …) | Logical predicates: `on(A, B)`, `clear(A)` |
| **Scalability** | Exponential blowup | Compact lifted representation |
| **Policies** | Per-state lookup table | Generalizable: "if *any* block is clear, pick it up" |

A **First-Order Decision Diagram (FODD)** is a DAG analogous to a BDD but over first-order atoms. pyrmdp uses FODDs for symbolic dynamic programming — automatically discovering and merging states that share the same value.

### 1.2 FODD Reduction Rules (R1–R5)

| Rule | Name | Description |
|------|------|-------------|
| **R1** | Neglect | High and low branches point to the same child → delete the node |
| **R2** | Join | Two nodes have identical test + children → merge (via unique table) |
| **R3** | Merge | Child re-tests a parent's predicate → follow the known branch |
| **R4** | Sort | Global predicate ordering ensures canonical structure |
| **R5** | Strong | Background knowledge (Z3) prunes UNSAT branches |

---

## 2. Synthesis Pipeline

The `pyrmdp.synthesis` subpackage implements a 6-step pipeline for **Domain Robustification & Symbolic Skill Synthesis**:

```
PDDL Domain ──► Step 1 ──► Step 2 ──► Step 3 ──► Step 4 ──► Step 5 ──► Step 6 ──► PPDDL
                 LLM        FODDs      SCC        MSCA       Delta       Multi-
                Failure     Abstract   Condens.   Bound      Minim.     Policy
                Halluc.     States                                      Emission
```

| Step | Module | What it does |
|------|--------|-------------|
| **1** | `llm_failure.py` | For each operator, query an LLM to hallucinate a plausible failure mode. Injects new predicates/types and failure effect branches into the domain. |
| **2** | `fodd_builder.py` | Bridge pyPPDDL `ActionSchema` objects → pyrmdp `FODDManager`. Compile precondition trees into FODDs, encode probabilistic effects as leaves, compose via `apply(max)`, enumerate leaf-paths as abstract state partitions → `nx.DiGraph`. |
| **3** | `graph_analysis.py` | Extract SCCs with `nx.condensation()` → DAG. Each DAG node is a set of abstract states that can already reach each other. |
| **4** | `graph_analysis.py` | Identify sources (in-deg = 0) and sinks (out-deg = 0). Compute the MSCA bound: min new edges = max(\|sources\|, \|sinks\|). |
| **5** | `delta_minimizer.py` | Iteratively pick the best sink→source pair (weighted scoring: α·(1−Δ) + β·gain), ask the LLM to synthesize a bridging operator + failure mode, add the edge, re-condense, repeat until irreducible. |
| **6** | `ppddl_emitter.py` | For every operator, emit K robot-policy variants (3 probabilistic branches: success/unchanged/worse with reward annotations) + 1 human-policy variant (deterministic, reward 0). |

**Dependency:** The pipeline uses [pyPPDDL](../POMDPDDL/pyPPDDL) for parsing and grounding PDDL/PPDDL domains. pyrmdp's own regex parser (`core.logic`) is retained for lightweight standalone usage.

---

## 3. Architecture

### 3.1 Project Structure

```
pyrmdp/
├── pyrmdp/
│   ├── __init__.py            # Public API façade
│   ├── core/
│   │   ├── logic.py           # Atom, Variable, Constant, PPDDL regex parser
│   │   ├── fodd.py            # FODDNode, FODDManager (unique table, global order)
│   │   └── markov.py          # AbstractTransitionMatrix, spectral gap,
│   │                          #   is_irreducible(), is_ergodic(),
│   │                          #   get_communicating_classes()
│   ├── pruning/
│   │   ├── reduction.py       # SyntacticReducer (R1, R3), apply(), StrongReducer (R5)
│   │   └── llm_axiom.py       # LLM → Z3 background knowledge generation
│   ├── synthesis/             # ← NEW: 6-step robustification pipeline
│   │   ├── __init__.py
│   │   ├── llm_failure.py     # Step 1 — LLM failure hallucination
│   │   ├── fodd_builder.py    # Step 2 — lifted FODD construction + abstract states
│   │   ├── graph_analysis.py  # Steps 3 & 4 — SCC condensation + MSCA bound
│   │   ├── delta_minimizer.py # Step 5 — iterative delta minimization
│   │   └── ppddl_emitter.py   # Step 6 — multi-policy PPDDL emission
│   └── vis/
│       └── visualization.py   # pyvis interactive FODD/graph plotting
├── scripts/
│   ├── generate_add.py        # Build & visualize an FODD from PPDDL
│   ├── generate_markov.py     # Build & visualize abstract Markov chain
│   └── test_pipeline.py       # End-to-end pipeline test (mock LLM, no API key)
└── setup.py
```

### 3.2 Core Modules

| Module | Key Classes / Functions |
|--------|----------------------|
| `core.logic` | `Variable`, `Constant`, `Atom`, `parse_ppddl_predicates()`, `parse_ppddl_actions()` |
| `core.fodd` | `FODDNode`, `FODDManager` (unique table, global order, `get_node()`, `get_leaf()`) |
| `core.markov` | `AbstractTransitionMatrix` — `build()`, `spectral_gap()`, `is_irreducible()`, `is_ergodic()`, `get_communicating_classes()` |
| `pruning.reduction` | `SyntacticReducer` (R1 + R3 with path-aware memoization), `apply(op, mgr, f1, f2)`, `StrongReducer` (Z3) |
| `pruning.llm_axiom` | `generate_background_knowledge(predicates)` → Z3 axiom strings |
| `vis.visualization` | `plot_fodd_structure(manager, root)` → interactive HTML |

### 3.3 Synthesis Modules

| Module | Key Classes / Functions |
|--------|----------------------|
| `synthesis.llm_failure` | `hallucinate_failures(domain, llm_fn=, failure_prob=)` → `(Domain, List[FailureHallucinationResult])` |
| `synthesis.fodd_builder` | `map_pyppddl_to_pyrmdp()`, `build_precondition_fodd()`, `build_effect_fodd()`, `build_transition_fodd()`, `enumerate_abstract_states()` → `nx.DiGraph` |
| `synthesis.graph_analysis` | `condense_to_dag()` → `CondensationResult`, `compute_augmentation_bound()` → `AugmentationBound` |
| `synthesis.delta_minimizer` | `delta_minimize(graph, domain, llm_fn=)` → `DeltaMinimizationResult`, `ScoringConfig`, `CandidateEdge` |
| `synthesis.ppddl_emitter` | `emit_ppddl(domain, output_path=, config=)` → PPDDL string, `PolicyExpansionConfig` |

---

## 4. Installation

```bash
git clone https://github.com/your-username/pyrmdp.git
cd pyrmdp

# Core (FODDs + synthesis pipeline)
pip install -e .

# With LLM support (OpenAI)
pip install -e ".[llm]"

# With Z3 strong reduction
pip install -e ".[z3]"

# Everything
pip install -e ".[all]"
```

The synthesis pipeline also requires **pyPPDDL** for domain parsing:

```bash
pip install -e /path/to/pyPPDDL
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy`, `scipy` | ✅ | Transition matrices, spectral analysis |
| `networkx` | ✅ | SCC condensation, graph algorithms |
| `pyvis` | ✅ | Interactive HTML visualization |
| `openai` | Optional | LLM failure hallucination & operator synthesis |
| `z3-solver` | Optional | Strong reduction (R5) |

---

## 5. Quick Start

### 5.1 FODD Visualization

```bash
# Build an FODD from a PPDDL domain
python scripts/generate_add.py

# Build an abstract Markov chain
python scripts/generate_markov.py
```

Both produce interactive `.html` files viewable in a browser.

### 5.2 Running the Synthesis Pipeline

**With mock LLM (no API key needed):**

```bash
python scripts/test_pipeline.py
```

This runs the full 6-step pipeline on a tabletop-grid-reward domain and prints progress for each step.

**Programmatic usage:**

```python
from pyppddl.ppddl.parser import load_domain
from pyrmdp.core.fodd import FODDManager
from pyrmdp.synthesis import (
    hallucinate_failures,
    build_transition_fodd,
    enumerate_abstract_states,
    condense_to_dag,
    compute_augmentation_bound,
    delta_minimize,
    emit_ppddl,
    ScoringConfig,
)
from pyrmdp.synthesis.fodd_builder import build_global_order

# Step 0: Parse
domain = load_domain("domain.pddl")

# Step 1: Hallucinate failures (pass llm_fn= for custom LLM)
domain, failures = hallucinate_failures(domain, failure_prob=0.1)

# Step 2: Build lifted FODDs → abstract state graph
manager = FODDManager(global_order=build_global_order(domain))
composed, action_fodds = build_transition_fodd(domain.actions, manager)
graph = enumerate_abstract_states(action_fodds, manager, domain)

# Steps 3–4: Condense + bound
condensation = condense_to_dag(graph)
bound = compute_augmentation_bound(condensation)
print(f"MSCA bound: {bound.bound}")

# Step 5: Synthesize recovery operators
result = delta_minimize(graph, domain, config=ScoringConfig(max_iterations=20))
print(f"Irreducible: {result.is_irreducible}")

# Step 6: Emit multi-policy PPDDL
ppddl = emit_ppddl(domain, output_path="robustified.ppddl")
```

---

## 6. References

1.  **Boutilier, C., Reiter, R., & Price, B. (2001).** Symbolic dynamic programming for first-order MDPs. *IJCAI*, 1, 690-700.
2.  **Wang, C., Joshi, S., & Khardon, R. (2008).** First order decision diagrams for relational MDPs. *Journal of Artificial Intelligence Research*, 31, 431-472.
3.  **Eswaran, K. P. & Tarjan, R. E. (1976).** Augmentation problems. *SIAM Journal on Computing*, 5(4), 653-665. *(MSCA theorem)*
