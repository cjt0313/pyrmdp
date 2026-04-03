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
RGB + Task NL ──► Step 0 ──►  ┌─ Step 1 ──► Step 2 ──► Δ check ──► Step 3 ──► Step 4 ──► Step 5 ─┐
                   Domain     │    LLM       FODDs      Spectral      SCC        MSCA       Delta  │
                   Genesis    │   Failure    Abstract   Distance    Condens.    Bound      Minim.  │
                  (VLM+LLM)   │   Halluc.    States     Δ < ε?                                    │
                              │                          │ yes                                     │
                              │      Converged? ◄────────┘           Loop back (new ops) ◄────────┘
                              └──────── no ──────────────────────────────────────────────────────────┘
                                        │ yes
                                        ▼
                                     Step 6 ──► PPDDL
                                     Multi-Policy Emission
```

| Step | Module | What it does |
|------|--------|-------------|
| **0a** | `prompts/vlm_domain_prompt.py` | Send RGB image(s) to a VLM → generates `:types` and `:predicates` (always includes a `robot` type). No operators at this stage. |
| **0b** | `prompts/llm_operator_prompt.py` | Send NL task descriptions + generated types/predicates to an LLM → generates `:action` blocks with preconditions and effects. |
| **0** | `domain_genesis.py` | Orchestrates 0a + 0b, assembles a complete PDDL domain file as initialization for Step 1. |
| **1** | `llm_failure.py` | For each operator, query an LLM to hallucinate a plausible failure mode. Injects new predicates/types and failure effect branches into the domain. |
| **2** | `fodd_builder.py` | Bridge pyPPDDL `ActionSchema` objects → pyrmdp `FODDManager`. Compile precondition trees into FODDs, encode probabilistic effects as leaves, compose via `apply(max)`, enumerate leaf-paths as abstract state partitions → `nx.DiGraph`. |
| **3** | `graph_analysis.py` | Extract SCCs with `nx.condensation()` → DAG. Each DAG node is a set of abstract states that can already reach each other. |
| **4** | `graph_analysis.py` | Identify sources (in-deg = 0) and sinks (out-deg = 0). Compute the MSCA bound: min new edges = max(\|sources\|, \|sinks\|). |
| **5** | `delta_minimizer.py` | Iteratively pick the best sink→source pair (weighted scoring: α·(1−Δ) + β·gain), ask the LLM to synthesize a bridging operator + failure mode, add the edge, re-condense, repeat until irreducible. |
| **Loop** | `iterative_synthesizer.py` | Wraps Steps 1–5 in a convergence-controlled while-loop. Each iteration hallucinate failures for new operators, rebuilds the abstract graph, extracts the transition matrix $M_{abs}$, and checks the **spectral distance** $\Delta = \|\Lambda_{\text{curr}} - \Lambda_{\text{prev}}\|_2$ (eigenvalue arrays zero-padded to equal length). Loop breaks when $\Delta < \varepsilon$. |
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
│   ├── synthesis/             # ← 7-step robustification pipeline (Step 0–6)
│   │   ├── __init__.py
│   │   ├── config.py          # PipelineConfig — all tuneable params, YAML I/O
│   │   ├── prompts/           # Prompt templates (one file per query purpose)
│   │   │   ├── vlm_domain_prompt.py   # Step 0a — RGB → types + predicates
│   │   │   └── llm_operator_prompt.py # Step 0b — task NL → operators
│   │   ├── domain_genesis.py  # Step 0 — orchestrates 0a + 0b → PDDL domain
│   │   ├── llm_failure.py     # Step 1 — LLM failure hallucination
│   │   ├── fodd_builder.py    # Step 2 — lifted FODD construction + abstract states
│   │   ├── graph_analysis.py  # Steps 3 & 4 — SCC condensation + MSCA bound
│   │   ├── delta_minimizer.py # Step 5 — iterative delta minimization
│   │   ├── iterative_synthesizer.py  # Iterative loop (Steps 1–5) w/ spectral convergence
│   │   └── ppddl_emitter.py   # Step 6 — multi-policy PPDDL emission
│   └── vis/
│       └── visualization.py   # pyvis interactive FODD/graph plotting
├── scripts/
│   ├── run_pipeline.py        # End-to-end pipeline (Steps 0–6) CLI
│   ├── generate_add.py        # Build & visualize an FODD from PPDDL
│   └── generate_markov.py     # Build & visualize abstract Markov chain
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
| `synthesis.prompts.vlm_domain_prompt` | `build_vlm_domain_prompt(scene_description=, domain_name_hint=)` → `{system, user}`, `parse_vlm_domain_response()` |
| `synthesis.prompts.llm_operator_prompt` | `build_llm_operator_prompt(domain_fragment, task_descriptions, extra_context=)` → `{system, user}`, `parse_llm_operator_response()` |
| `synthesis.domain_genesis` | `generate_initial_domain(image_paths=, task_descriptions=, vlm_fn=, llm_fn=, return_parsed=)` → PDDL str or `Domain` |
| `synthesis.llm_failure` | `hallucinate_failures(domain, llm_fn=, failure_prob=)` → `(Domain, List[FailureHallucinationResult])` |
| `synthesis.fodd_builder` | `map_pyppddl_to_pyrmdp()`, `build_precondition_fodd()`, `build_effect_fodd()`, `build_transition_fodd()`, `enumerate_abstract_states()` → `nx.DiGraph` |
| `synthesis.graph_analysis` | `condense_to_dag()` → `CondensationResult`, `compute_augmentation_bound()` → `AugmentationBound` |
| `synthesis.delta_minimizer` | `delta_minimize(graph, domain, llm_fn=)` → `DeltaMinimizationResult`, `ScoringConfig`, `CandidateEdge` |
| `synthesis.iterative_synthesizer` | `IterativeDomainSynthesizer(domain, epsilon=, max_iterations=, scoring_config=, emission_config=)` — iterative Steps 1–5 loop + Step 6 emission with spectral-distance convergence; `IterationRecord` per-iteration diagnostics; `extract_transition_matrix()`, `compute_sorted_eigenvalues()`, `spectral_distance()` utilities |
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

### 5.1 End-to-End Pipeline (`run_pipeline.py`)

The main entry point is `scripts/run_pipeline.py`. It runs **Steps 0–6** from a single command.

**From RGB image + task description (full pipeline):**

```bash
python scripts/run_pipeline.py \
    --image scene.png \
    --task "Pick up the red block and place it on the blue plate." \
    --num-policies 3 \
    --output-dir ./pipeline_output
```

**Multiple images and task sentences:**

```bash
python scripts/run_pipeline.py \
    --image scene_front.png --image scene_top.png \
    --task "Stack all blocks on the tray." \
    --task "If a block falls, pick it up and retry." \
    -K 3 -o ./pipeline_output
```

**From an existing PDDL domain (skip Step 0):**

```bash
python scripts/run_pipeline.py \
    --domain domain.pddl \
    --num-policies 3 \
    --output-dir ./pipeline_output
```

**With intermediate results saved alongside the final PPDDL:**

```bash
python scripts/run_pipeline.py \
    --image scene.png \
    --task "Stack blocks." \
    -K 3 -o ./pipeline_output \
    --save-intermediates -v
```

**Using a YAML config file (recommended for reproducibility):**

```bash
# Generate a default config template
python scripts/run_pipeline.py --dump-config -o ./my_experiment

# Edit ./my_experiment/pipeline.yaml, then:
python scripts/run_pipeline.py \
    --domain domain.pddl \
    --config ./my_experiment/pipeline.yaml -v

# CLI flags override config values:
python scripts/run_pipeline.py \
    --domain domain.pddl \
    --config pipeline.yaml \
    --epsilon 0.01 --num-policies 5
```

#### Configuration (`pipeline.yaml`)

All technical parameters live in a single `PipelineConfig` dataclass
(`pyrmdp/synthesis/config.py`) that can be serialized to/from YAML.
Generate a default template:

```bash
python -c "from pyrmdp.synthesis.config import PipelineConfig; PipelineConfig.write_default('pipeline.yaml')"
# or:
python scripts/run_pipeline.py --dump-config
```

This produces a `pipeline.yaml` with every tunable parameter, grouped by step:

```yaml
# pyrmdp pipeline configuration

# ── Iterative convergence loop
epsilon: 0.05               # Δ_spectral < ε → stop
max_loop_iterations: 10     # hard cap on outer loop

# ── Step 1: Failure Hallucination
failure_prob: 0.1           # P(failure branch) per action

# ── Step 5: Delta Minimization
scoring_alpha: 0.7          # weight for delta similarity
scoring_beta: 0.3           # weight for topological gain
max_delta_iterations: 50    # max synthesis iterations per loop pass
max_candidates_per_iter: 10
delta_threshold: 15         # max predicate delta for LLM prompt

# ── Step 6: Multi-Policy Emission
num_robot_policies: 3
success_reward: 10.0
unchanged_reward: -1.0
failure_reward: -10.0
human_reward: 0.0
initial_success_prob: 0.333
initial_unchanged_prob: 0.333
initial_failure_prob: 0.333

# ── Output
output_dir: ./pipeline_output
save_intermediates: false
```

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image`, `-i` | — | RGB image path (repeatable for multiple views) |
| `--task`, `-t` | — | NL task description (repeatable) |
| `--domain`, `-d` | — | Existing PDDL file (skips Step 0) |
| `--config`, `-c` | — | YAML config file; CLI flags override its values |
| `--dump-config` | — | Write a default `pipeline.yaml` to `--output-dir` and exit |
| `--output-dir`, `-o` | `./pipeline_output` | Directory for all outputs (final PPDDL + intermediates) |
| `--num-policies`, `-K` | `3` | Robot policy variants per action |
| `--save-intermediates` | off | Save per-step JSON/GraphML files to output dir |
| `--failure-prob` | `0.1` | Failure branch probability (Step 1) |
| `--max-delta-iterations` | `50` | Max synthesis iterations (Step 5) |
| `--scoring-alpha` | `0.7` | Delta similarity weight (Step 5) |
| `--scoring-beta` | `0.3` | Topological gain weight (Step 5) |
| `--epsilon` | `0.05` | Spectral-distance convergence threshold for the iterative loop |
| `--max-loop-iterations` | `10` | Maximum iterations for the Steps 1–5 convergence loop |
| `--success-reward` | `10.0` | Reward for success branch (Step 6) |
| `--unchanged-reward` | `-1.0` | Reward for unchanged branch (Step 6) |
| `--failure-reward` | `-10.0` | Reward for failure branch (Step 6) |
| `--human-reward` | `0.0` | Reward for human policy (Step 6) |
| `-v` | off | Verbose (DEBUG) logging |

#### Intermediate Outputs (`--save-intermediates`)

When enabled, the following files are saved alongside `robustified.ppddl` in the output directory. Each iteration of the convergence loop produces its own set of `iterN_*` files:
| File | Step | Content |
|------|------|--------|
| `step0_domain_generated.pddl` | 0 | PDDL domain from VLM + LLM |
| `iter{N}_step1_failures.json` | 1 | Hallucinated failure modes per action (iteration N) |
| `iter{N}_step2_abstract_graph.graphml` | 2 | Abstract transition graph (iteration N) |
| `iter{N}_step2_abstract_states.json` | 2 | All abstract states with predicate assignments |
| `iter{N}_step3_condensation.json` | 3 | SCC decomposition |
| `iter{N}_step3_dag.graphml` | 3 | Condensation DAG |
| `iter{N}_step4_augmentation_bound.json` | 4 | Sources, sinks, MSCA bound |
| `iter{N}_step5_synthesized_operators.json` | 5 | Synthesized recovery operators |
| `iter{N}_step5_stats.json` | 5 | Delta minimization statistics |
| `pipeline_summary.json` | — | Convergence diagnostics, timings, spectral distances |

### 5.2 FODD Visualization

```bash
python scripts/generate_add.py
python scripts/generate_markov.py
```

Both produce interactive `.html` files viewable in a browser.

### 5.3 Programmatic Usage

**Using `PipelineConfig` (recommended):**

```python
from pyppddl.ppddl.parser import load_domain
from pyrmdp.synthesis.config import PipelineConfig
from pyrmdp.synthesis.iterative_synthesizer import IterativeDomainSynthesizer

# All params in one place — edit or load from YAML
cfg = PipelineConfig(
    epsilon=0.01,
    max_loop_iterations=15,
    failure_prob=0.1,
    scoring_alpha=0.7,
    scoring_beta=0.3,
    num_robot_policies=5,
    save_intermediates=True,
    output_dir="./my_experiment",
)
# Or load from a YAML file:
# cfg = PipelineConfig.from_yaml("pipeline.yaml")

domain = load_domain("domain.pddl")

synth = IterativeDomainSynthesizer(
    domain,
    epsilon=cfg.epsilon,
    max_iterations=cfg.max_loop_iterations,
    failure_prob=cfg.failure_prob,
    scoring_config=cfg.scoring_config(),   # → ScoringConfig
    emission_config=cfg.emission_config(), # → PolicyExpansionConfig
    output_dir=cfg.output_dir,
    save_intermediates=cfg.save_intermediates,
)

ppddl_str = synth.run()            # Steps 1–5 loop + Step 6 emission

print(synth.summary())             # {"converged": true, "iterations": 3, ...}
cfg.write_yaml("./my_experiment/pipeline.yaml")  # save for repro
```

**Full pipeline from Python (Step 0 → iterative loop → PPDDL):**

```python
from pyrmdp.synthesis.config import PipelineConfig
from scripts.run_pipeline import run_pipeline

cfg = PipelineConfig.from_yaml("pipeline.yaml")  # or PipelineConfig(...)

ppddl = run_pipeline(
    image_paths=["scene.png"],
    task_descriptions=["Pick up the red block and place it on the blue plate."],
    config=cfg,
)
```

---

## 6. References

1.  **Boutilier, C., Reiter, R., & Price, B. (2001).** Symbolic dynamic programming for first-order MDPs. *IJCAI*, 1, 690-700.
2.  **Wang, C., Joshi, S., & Khardon, R. (2008).** First order decision diagrams for relational MDPs. *Journal of Artificial Intelligence Research*, 31, 431-472.
3.  **Eswaran, K. P. & Tarjan, R. E. (1976).** Augmentation problems. *SIAM Journal on Computing*, 5(4), 653-665. *(MSCA theorem)*
