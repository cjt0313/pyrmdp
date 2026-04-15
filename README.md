# pyrmdp — Lifted FODDs + Domain Robustification Pipeline

`pyrmdp` is a Python library for **Relational Markov Decision Processes (RMDPs)** built on **First-Order Decision Diagrams (FODDs)**. It works directly with the **lifted** (relational) representation — no exponential grounding required.

On top of the FODD core, pyrmdp ships a **synthesis pipeline** that:

1. Generates a PDDL domain from a single RGB image + natural-language task description,
2. Hallucinates failure modes, extracts an abstract Markov chain via lifted FODDs,
3. Deterministically synthesizes recovery operators (via variable unification) until the chain is strongly connected,
4. Emits a multi-policy PPDDL with reward-annotated human/robot branches.

**TL;DR** — Given an image + task → output a complete PPDDL domain.

---

## Table of Contents

- [1. Core: FODDs & Reductions](#1-core-fodds--reductions)
  - [1.1 Relational MDPs](#11-relational-mdps)
  - [1.2 FODD Reduction Rules (R1–R5)](#12-fodd-reduction-rules-r1r5)
- [2. Synthesis Pipeline](#2-synthesis-pipeline)
  - [2.1 Pipeline Overview](#21-pipeline-overview)
  - [2.2 Step 0 — Domain Genesis (VLM + LLM)](#22-step-0--domain-genesis-vlm--llm)
  - [2.3 Step 1 — LLM Failure Hallucination](#23-step-1--llm-failure-hallucination)
  - [2.4 Step 2 — Lifted FODD Construction & Abstract State Enumeration](#24-step-2--lifted-fodd-construction--abstract-state-enumeration)
  - [2.5 R5 — LLM-Based Mutex Pruning (Optional)](#25-r5--llm-based-mutex-pruning-optional)
  - [2.6 Step 3 — SCC Condensation](#26-step-3--scc-condensation)
  - [2.7 Step 4 — MSCA Bound Computation](#27-step-4--msca-bound-computation)
  - [2.8 Step 5 — Delta Minimization & Deterministic Recovery Synthesis](#28-step-5--delta-minimization--deterministic-recovery-synthesis)
  - [2.9 Spectral Convergence Loop (Steps 1–5)](#29-spectral-convergence-loop-steps-15)
  - [2.10 Step 6 — Multi-Policy PPDDL Emission](#210-step-6--multi-policy-ppddl-emission)
- [3. Architecture](#3-architecture)
  - [3.1 Project Structure](#31-project-structure)
  - [3.2 Prompt Architecture](#32-prompt-architecture)
  - [3.3 Configuration System](#33-configuration-system)
- [4. Installation](#4-installation)
- [5. Quick Start](#5-quick-start)
  - [5.1 End-to-End Pipeline](#51-end-to-end-pipeline)
  - [5.2 Configuration](#52-configuration)
  - [5.3 CLI Reference](#53-cli-reference)
  - [5.4 Programmatic Usage](#54-programmatic-usage)
  - [5.5 FODD Visualization](#55-fodd-visualization)
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
| **R5** | Strong | Background knowledge (Z3 or LLM mutex rules) prunes UNSAT branches |

---

## 2. Synthesis Pipeline

### 2.1 Pipeline Overview

```
RGB + Task NL ──► Step 0 ──►  ┌─ Step 1 ──► Step 2 ──► (R5) ──► Δ check ──► Step 3 ──► Step 4 ──► Step 5 ─┐
                   Domain     │    LLM       FODDs     Mutex     Spectral      SCC        MSCA       Delta  │
                   Genesis    │   Failure    Abstract  Pruning   Distance    Condens.    Bound      Minim.  │
                  (VLM+LLM)   │   Halluc.    States   (optional) Δ < ε?                                    │
                              │                                   │ yes                                     │
                              │      Converged? ◄─────────────────┘           Loop back (new ops) ◄────────┘
                              └──────── no ──────────────────────────────────────────────────────────────────┘
                                        │ yes
                                        ▼
                                     Step 6 ──► PPDDL
                                     Multi-Policy Emission
```

The pipeline has **4 LLM-based tasks**, each with an independent prompt file.
Step 5 (recovery synthesis) is **fully deterministic** — no LLM call required.

| LLM Task | Prompt File | Step |
|----------|-------------|------|
| Scene → Types + Predicates (VLM) | `prompts/vlm_domain_prompt.py` | 0a |
| Task NL → Operators (LLM) | `prompts/llm_operator_prompt.py` | 0b |
| Action → Failure Mode (LLM) | `prompts/llm_failure_prompt.py` | 1 |
| Predicates → Mutex Constraints (LLM) | `prompts/llm_mutex_prompt.py` | R5 |

> **Note:** Step 5 previously used `prompts/llm_recovery_prompt.py` to query the
> LLM for recovery operators. This was replaced with deterministic synthesis
> (variable unification + minimal causal preconditions) — see §2.8.

---

### 2.2 Step 0 — Domain Genesis (VLM + LLM)

**Module:** `synthesis/domain_genesis.py`

Generates a complete PDDL domain from scratch using a two-phase approach.

**Phase A — VLM Scene Analysis:**

| | |
|---|---|
| **Input** | One or more RGB images of the scene + optional scene description |
| **LLM Task** | Vision-Language Model query |
| **Prompt** | `prompts/vlm_domain_prompt.py` |
| **Output** | PDDL `:types` and `:predicates` block (always includes a `robot` type) |

The VLM is instructed to generate predicates from **three perspectives**:
1. **Object-self** — intrinsic properties (e.g. `opened`, `upright`, `graspable`)
2. **Object-object** — relations between objects (e.g. `on`, `inside`, `holding`)
3. **Object-environment** — workspace relations (e.g. `on-table`, `clear`, `arm-empty`)

**Phase B — LLM Operator Generation:**

| | |
|---|---|
| **Input** | Generated types/predicates + natural-language task descriptions |
| **LLM Task** | Text-only LLM query |
| **Prompt** | `prompts/llm_operator_prompt.py` |
| **Output** | PDDL `:action` blocks (deterministic, STRIPS-compatible) |

The LLM generates **only** operators that are directly specified or directly implied by the task sentences. Each action must be traceable to a specific phrase or step in the task description — no extra "generally useful" manipulation primitives are added. For example, given *"close the box"* and *"pick up the marker and put it into another cup"*, the LLM produces operators for close, pick-up, and put-in — but not unrelated skills like stack or place-on-surface.

The two phases are composed into a single `(define (domain ...) ...)` block. If the LLM generates 0 actions, a retry is triggered asking it to re-read the task sentences.

**Final Output:** Complete PDDL domain string (or `GenesisResult` with parsed `Domain` + VLM origin tracking).

---

### 2.3 Step 1 — LLM Failure Hallucination

**Module:** `synthesis/llm_failure.py`

For each operator in the domain, query the LLM to hallucinate a physically plausible failure mode ("worse effect").

| | |
|---|---|
| **Input** | Parsed PDDL `Domain` with action schemas |
| **LLM Task** | Per-action failure-mode generation |
| **Prompt** | `prompts/llm_failure_prompt.py` |
| **Output** | Augmented `Domain` with probabilistic failure branches |

**Algorithm:**
1. For each `ActionSchema` in the domain:
   - Serialize the action (parameters, precondition, effects, domain context) into a human-readable description
   - Query the LLM for a JSON response containing `failure_add`, `failure_del`, `failure_numeric`, `new_predicates`, `new_types`
2. Parse the response and inject:
   - New predicates/types into the domain (if any)
   - A failure-branch `Effect` with probability `failure_prob` (default 0.1)
   - Rescale existing effects to `(1 - failure_prob)`

**Output format per action:**
```
Original effect (prob: 0.9) → nominal success
New failure effect (prob: 0.1) → hallucinated failure
```

---

### 2.4 Step 2 — Lifted FODD Construction & Abstract State Enumeration

**Module:** `synthesis/fodd_builder.py`

Bridge `pyPPDDL.ActionSchema` objects into pyrmdp's `FODDManager`. Compile preconditions and probabilistic effects into FODDs and enumerate abstract state partitions.

| | |
|---|---|
| **Input** | Augmented `Domain` (with failure branches) |
| **Output** | `nx.DiGraph` — abstract transition graph |

**Algorithm:**
1. Map pyPPDDL types/predicates to a global FODD predicate ordering
2. For each action, build:
   - **Precondition FODD** — encodes when the action is applicable
   - **Effect FODD** — encodes probabilistic outcomes (success/failure leaves)
   - **Transition FODD** — composed via `apply(max)`
3. Enumerate leaf-paths as abstract state partitions:
   - Each path through the FODD defines a conjunction of predicate truth assignments
   - States are represented as `(true_predicates: frozenset, false_predicates: frozenset)`
4. Build a directed graph where edges represent possible transitions with their probabilities

---

### 2.5 R5 — LLM-Based Mutex Pruning (Optional)

**Module:** `pruning/llm_axiom.py`

When `--enable-mutex-pruning` is set, query the LLM for domain mutex constraints and use them to prune impossible abstract states.

| | |
|---|---|
| **Input** | Abstract transition graph + predicate names |
| **LLM Task** | Mutex constraint generation |
| **Prompt** | `prompts/llm_mutex_prompt.py` |
| **Output** | Pruned `nx.DiGraph` (fewer nodes) |

**Three kinds of constraints:**
- **positive_mutex**: pred_a and pred_b cannot both be TRUE (e.g. `holding` ∧ `arm-empty`)
- **negative_mutex**: pred_a and pred_b cannot both be FALSE (at least one must hold)
- **implication**: If pred_a is TRUE then pred_b must be TRUE (e.g. `holding` → `graspable`)

**Algorithm:**
1. Query the LLM with the domain's predicate names
2. Parse and validate rules (drop rules referencing unknown predicates)
3. For each abstract state in the graph:
   - Check all mutex rules against the state's truth assignment
   - If any rule is violated → remove the state (and its edges) from the graph

---

### 2.6 Step 3 — SCC Condensation

**Module:** `synthesis/graph_analysis.py` → `condense_to_dag()`

Extract Strongly Connected Components (SCCs) from the abstract transition graph and collapse each SCC into a single DAG node.

| | |
|---|---|
| **Input** | Abstract transition graph (`nx.DiGraph`) |
| **Output** | `CondensationResult` — a DAG of SCCs |

**Algorithm:**
1. Compute SCCs via `nx.condensation()` → DAG
2. Each DAG node is a set of abstract states that can already reach each other (through existing transitions)
3. Record the SCC → member-state mapping and representative predicate assignments

---

### 2.7 Step 4 — MSCA Bound Computation

**Module:** `synthesis/graph_analysis.py` → `compute_augmentation_bound()`

Identify structural deficiencies in the condensation DAG.

| | |
|---|---|
| **Input** | `CondensationResult` (DAG of SCCs) |
| **Output** | `AugmentationBound` — sources, sinks, minimum new edges |

**Algorithm:**
1. Identify **sources** (in-degree = 0 in the DAG) and **sinks** (out-degree = 0)
2. Compute the **MSCA bound**: min edges = max(|sources|, |sinks|) — based on the Eswaran-Tarjan (1976) theorem
3. If sources = sinks = ∅ → the graph is already irreducible (strongly connected)

---

### 2.8 Step 5 — Delta Minimization & Deterministic Recovery Synthesis

**Module:** `synthesis/delta_minimizer.py`

Synthesize recovery operators to make the abstract graph strongly connected (irreducible). Recovery operators are computed **deterministically** from the predicate delta — no LLM call required. The algorithm uses a **monotonic cycle-closing loop** with **Minimum Weight Cycle Cover (MWCC)** for inter-component routing and **reachability-based cycle closure** for intra-component merging.

| | |
|---|---|
| **Input** | Abstract graph, `Domain` |
| **LLM Task** | *None* — fully deterministic |
| **Output** | `DeltaMinimizationResult` — synthesized operators + updated graph |

#### Algorithm 1 — Main Augmentation Loop (`_compute_augmentation_edges`)

A monotonic `while True` loop that terminates when a single SCC remains:

1. Build `H_raw = G + accumulated_meta_edges`, condense to `H = nx.condensation(H_raw)`
2. If `|H.nodes| == 1` → done
3. If `|WCCs| > 1` → call MWCC (Algorithm 2) to merge WCCs
4. If `|WCCs| == 1` → call Reachability Closure (Algorithm 3) to merge SCCs
5. Repeat

**Monotonic convergence:** Each iteration reduces either `|WCCs|` or `|SCCs|` by ≥ 1, so the loop terminates in at most `|V|` steps.

#### Algorithm 2 — Inter-WCC MWCC (`_build_global_cycle_across_wccs`)

*Goal:* Merge all `k` WCCs into a single weakly connected graph at minimum total Hamming cost.

1. Extract sinks/sources per WCC
2. Build a `k×k` cost matrix `C[i,j]` = min-distance bridge from a sink of WCC_i to a source of WCC_j (∞ on diagonal)
3. Solve with `scipy.optimize.linear_sum_assignment(C)` → minimum-weight successor permutation (Hungarian algorithm, O(k³))
4. **Cycle patching:** merge disjoint permutation cycles into one Hamiltonian cycle by swapping successors
5. Return witness edges `(u*, v*)` for each arc in the global cycle

#### Algorithm 3 — Intra-WCC Reachability Closure (`_close_one_reachable_source_sink_cycle`)

*Goal:* Reduce the SCC count by 1 within a single WCC.

1. Find all sources (in-degree 0) and sinks (out-degree 0) in the condensation `H`
2. For each source `s`, compute `nx.descendants(H, s)` to find reachable sinks
3. For each valid `(s, t)` pair where `s` reaches `t`, compute `_best_original_bridge(t, s)` — the edge goes FROM sink TO source (closing a directed cycle)
4. Pick the `(t, s)` pair with absolute minimum Hamming distance
5. Return that single witness edge

#### Operator Synthesis (`_lift_to_operator`)

For each `(sink_state, source_state)` pair returned by the augmentation loop:
- **Effects** — `must_add = source_true − sink_true`, `must_del = sink_true − source_true`
- **Variable Unification** — each bare predicate name is looked up in the domain's `:predicates` and grounded using the domain's canonical variable names
- **Minimal Causal Preconditions** — only sink-state predicates that share ≥ 1 unified variable with the delta effects
- **Single deterministic effect** — probability 1.0, no failure branch, no numeric rewards

**Logical Hamming Distance** counts predicates that must change truth value:
```
Δ(U, V) = |T_U ∩ F_V| + |F_U ∩ T_V| + |T_V \ (T_U ∪ F_U)| + |F_V \ (T_U ∪ F_U)|
```

**Stats output** (`iter*_step5_stats.json`): `phase1_ops` (MWCC edges), `phase2_ops` (reachability edges), `num_wccs`.

**Why MWCC?** The Hungarian algorithm finds the minimum-cost perfect matching in O(k³), provably optimal for the inter-component routing subproblem. The cycle patching step merges any disjoint permutation cycles into a single Hamiltonian cycle over WCCs.

**Why reachability closure?** By only adding edges where the source already reaches the sink in the DAG, each added edge is guaranteed to close a directed cycle and merge ≥ 2 SCCs into 1 — no wasted edges.

**Why deterministic?** The precondition and effect are fully determined by the abstract state pair — there is nothing for the LLM to invent. Step 5 runs **instantly** (no HTTP requests), making the overall pipeline significantly faster.

---

### 2.9 Spectral Convergence Loop (Steps 1–5)

**Module:** `synthesis/iterative_synthesizer.py`

Wraps Steps 1–5 in a convergence-controlled while-loop.

| | |
|---|---|
| **Input** | Initial `Domain`, `PipelineConfig` |
| **Output** | Final PPDDL string + convergence diagnostics |

**Algorithm:**
1. Run Steps 1–5 (hallucinate failures → build graph → condense → augment)
2. Extract the transition matrix M_abs from the abstract graph
3. Compute sorted eigenvalue magnitude arrays Λ_curr and Λ_prev
4. Measure **spectral distance** via three metrics (see below)
5. If Δ_Wasserstein < ε → converged; else loop back to Step 1 with the updated domain
6. On convergence → proceed to Step 6

**Spectral Distance Metrics:**

Three metrics are computed each iteration for convergence checking and ablation analysis:

| Metric | Formula | Range | Dimension-invariant? | Role |
|--------|---------|-------|---------------------|------|
| **Wasserstein distance** (primary) | W₁(Λ_curr, Λ_prev) | [0, ∞) | ✅ (no padding needed) | **Convergence criterion** (Δ < ε) |
| **Cosine distance** (secondary) | 1 − cos(Λ_curr, Λ_prev) | [0, 2] | ✅ (zero-padded) | Ablation / diagnostics |
| **L2 norm** (secondary) | ‖Λ_curr − Λ_prev‖₂ | [0, ∞) | ❌ (grows with √n) | Baseline / ablation |

**Why Wasserstein (primary)?** `scipy.stats.wasserstein_distance` treats the sorted eigenvalue magnitudes
as empirical distributions and computes the optimal transport cost. It is naturally
dimension-invariant (no zero-padding needed) and provides the tightest convergence signal —
typically 1–2 orders of magnitude smaller than cosine distance. Default ε = 0.1.

**Why cosine (secondary)?** As the graph grows via recovery operators, new absorbing sinks add eigenvalue 1.
Raw L2 measures √(new_states) ≈ 2–3 per iteration regardless of structural convergence.
Cosine normalises by vector magnitude, making it insensitive to proportional growth.
Retained as a diagnostic metric alongside L2.

**Budget cap:** `max_recovery_per_iter` limits the number of recovery operators
synthesised per iteration (Step 5). With budget=1, each iteration adds at most
one operator, creating a longer convergence trajectory ideal for studying spectral
behaviour. The default is unlimited (all necessary operators per iteration).

**Convergence diagnostics** recorded in `pipeline_summary.json`:
- `spectral_distances_wasserstein` — Wasserstein EMD per iteration (primary)
- `spectral_distances` — cosine distance per iteration
- `spectral_distances_l2` — L2 norm per iteration
- `per_iteration[i].eigenvalues` — full eigenvalue magnitude array

---

### 2.10 Step 6 — Multi-Policy PPDDL Emission

**Module:** `synthesis/ppddl_emitter.py`

For each operator in the final domain, emit K robot-policy variants with probabilistic branches and reward annotations, plus a deterministic human-policy variant.

| | |
|---|---|
| **Input** | Final augmented `Domain`, `PolicyExpansionConfig` |
| **Output** | PPDDL string (written to `robustified.ppddl`) |

**Per-action output:**
```
;; Robot policy variant k (k = 1..K)
(:action action-name_robot_k
  :parameters (...)
  :precondition (...)
  :effect (probabilistic
    p_success (and <nominal_effects> (reward 10.0))
    p_unchanged (reward -1.0)
    p_failure (and <failure_effects> (reward -10.0))
  )
)

;; Human policy (deterministic, reward 0)
(:action action-name_human
  :parameters (...)
  :precondition (...)
  :effect (and <nominal_effects> (reward 0.0))
)
```

---

## 3. Architecture

### 3.1 Project Structure

```
pyrmdp/
├── pyrmdp/
│   ├── __init__.py                      # Public API façade
│   ├── core/
│   │   ├── logic.py                     # Atom, Variable, Constant, PPDDL regex parser
│   │   ├── fodd.py                      # FODDNode, FODDManager (unique table, global order)
│   │   └── markov.py                    # AbstractTransitionMatrix, spectral gap,
│   │                                    #   is_irreducible(), is_ergodic()
│   ├── pruning/
│   │   ├── reduction.py                 # SyntacticReducer (R1, R3), StrongReducer (R5/Z3)
│   │   └── llm_axiom.py                # R5 mutex generation + abstract state pruning
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── config.py                    # PipelineConfig — all tuneable params, YAML I/O
│   │   ├── llm_config.py               # LLMConfig — model connection params, llm.yaml
│   │   ├── prompts/                     # ★ One file per LLM task (see §3.2)
│   │   │   ├── __init__.py
│   │   │   ├── response_parser.py       # Shared JSON extraction utility
│   │   │   ├── vlm_domain_prompt.py     # Step 0a — RGB → types + predicates
│   │   │   ├── llm_operator_prompt.py   # Step 0b — task NL → operators
│   │   │   ├── llm_failure_prompt.py    # Step 1  — failure hallucination
│   │   │   ├── llm_recovery_prompt.py   # (legacy — Step 5 is now deterministic)
│   │   │   └── llm_mutex_prompt.py      # R5      — mutex constraint generation
│   │   ├── domain_genesis.py            # Step 0 — orchestrates 0a + 0b → PDDL domain
│   │   ├── llm_failure.py               # Step 1 — LLM failure hallucination logic
│   │   ├── fodd_builder.py              # Step 2 — lifted FODD construction + abstract states
│   │   ├── graph_analysis.py            # Steps 3 & 4 — SCC condensation + MSCA bound
│   │   ├── delta_minimizer.py           # Step 5 — deterministic recovery synthesis (no LLM)
│   │   ├── iterative_synthesizer.py     # Iterative loop (Steps 1–5) w/ spectral convergence
│   │   └── ppddl_emitter.py            # Step 6 — multi-policy PPDDL emission
│   └── vis/
│       └── visualization.py             # pyvis interactive FODD/graph plotting
├── scripts/
│   ├── run_pipeline.py                  # End-to-end pipeline (Steps 0–6) CLI
│   ├── run_all_testdata.py              # Batch runner — all test cases in parallel
│   ├── run_experiment1_convergence.py   # Experiment 1 — spectral convergence budget sweep
│   ├── visualize_evolution.py           # Interactive HTML evolution graph (vis.js)
│   ├── plot_convergence.py              # Publication-quality convergence plots
│   ├── plot_metrics_comparison.py       # 3-metric (cosine/L2/Wasserstein) comparison plot
│   ├── generate_add.py                  # Build & visualize an FODD from PPDDL
│   └── generate_markov.py              # Build & visualize abstract Markov chain
├── llm.yaml                             # LLM connection config (API key, model, etc.)
└── setup.py
```

### 3.2 Prompt Architecture

Every LLM-based task has its **own independent prompt file** in `pyrmdp/synthesis/prompts/`. Each file follows the same pattern:

```python
# Module docstring explaining the task

SYSTEM_PROMPT = """..."""          # Role & rules for the LLM
USER_PROMPT_TEMPLATE = """..."""   # Task-specific template with {placeholders}

def build_*_prompt(...) -> dict:   # Returns {"system": str, "user": str}
def parse_*_response(...) -> ...:  # Parses raw LLM text → structured data
```

All prompt files share `response_parser.py` for JSON extraction (strip markdown fences, fix trailing commas, locate `{…}` body, `json.loads()`).

**Separation of concerns:**
- **Prompt files** (`prompts/*.py`) — prompt templates + response parsing (pure text → data)
- **Logic files** (`llm_failure.py`, `delta_minimizer.py`, etc.) — domain objects, algorithm logic, LLM call orchestration

> **Note:** `delta_minimizer.py` no longer calls the LLM. Recovery operators
> are computed deterministically (see §2.8). The legacy prompt file
> `prompts/llm_recovery_prompt.py` is retained but unused.

### 3.3 Configuration System

pyrmdp uses **two config files** that separate concerns:

| Config | File | Class | Purpose |
|--------|------|-------|---------|
| **LLM Config** | `llm.yaml` | `LLMConfig` | Model connection: API key, base URL, model name, temperature, max tokens, timeout |
| **Pipeline Config** | `pipeline.yaml` | `PipelineConfig` | Algorithm parameters: ε, failure prob, scoring weights, K policies, rewards |

Both support YAML serialization and env-var overrides (`PYRMDP_*` prefix for LLM config, CLI flags for pipeline config).

---

## 4. Installation

```bash
git clone https://github.com/your-username/pyrmdp.git
cd pyrmdp

# Core (FODDs + synthesis pipeline)
pip install -e .

# With LLM support (OpenAI-compatible API)
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
| `pyyaml` | ✅ | YAML config loading |
| `openai` | Optional | LLM/VLM API calls |
| `z3-solver` | Optional | Strong reduction (R5 with Z3) |

---

## 5. Quick Start

### 5.1 End-to-End Pipeline

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

**With R5 mutex pruning enabled:**

```bash
python scripts/run_pipeline.py \
    --image scene.png \
    --task "Stack blocks." \
    --enable-mutex-pruning \
    --save-intermediates -v -o ./pipeline_output
```

### 5.2 Configuration

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

**`pipeline.yaml` example:**

```yaml
# pyrmdp pipeline configuration

# ── Iterative convergence loop
epsilon: 0.02               # Δ_cosine < ε → stop
max_loop_iterations: 10     # hard cap on outer loop
max_recovery_per_iter: null # budget cap per iteration (null = unlimited)

# ── Step 1: Failure Hallucination
failure_prob: 0.1           # P(failure branch) per action

# ── Step 2: Abstract State Pruning
enable_mutex_pruning: false  # enable R5 LLM mutex pruning

# ── Step 5: Delta Minimization (deterministic — no LLM)
scoring_alpha: 0.7          # weight for delta similarity
scoring_beta: 0.3           # weight for topological gain
max_delta_iterations: 50    # max synthesis iterations per loop pass
max_candidates_per_iter: 10

# ── Step 6: Multi-Policy Emission
num_robot_policies: 3
success_reward: 10.0
unchanged_reward: -1.0
failure_reward: -10.0
human_reward: 0.0

# ── Output
output_dir: ./pipeline_output
save_intermediates: false
```

**`llm.yaml` example:**

```yaml
provider: openai
api_key: sk-...              # or set OPENAI_API_KEY env var
base_url: https://api.openai.com/v1
model: gpt-4o
temperature: 0.7
max_tokens: 16384
timeout: 120.0
max_retries: 2
```

### 5.3 CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--image`, `-i` | — | RGB image path (repeatable for multiple views) |
| `--task`, `-t` | — | NL task description (repeatable) |
| `--domain`, `-d` | — | Existing PDDL file (skips Step 0) |
| `--config`, `-c` | — | YAML config file; CLI flags override its values |
| `--dump-config` | — | Write a default `pipeline.yaml` to `--output-dir` and exit |
| `--output-dir`, `-o` | `./pipeline_output` | Directory for all outputs |
| `--num-policies`, `-K` | `3` | Robot policy variants per action (Step 6) |
| `--enable-mutex-pruning` | off | Enable R5 LLM-based mutex pruning (Step 2) |
| `--save-intermediates` | off | Save per-step JSON/GraphML files |
| `--failure-prob` | `0.1` | Failure branch probability (Step 1) |
| `--max-delta-iterations` | `50` | Max synthesis iterations — Step 5 (deterministic) |
| `--scoring-alpha` | `0.7` | Delta similarity weight (Step 5) |
| `--scoring-beta` | `0.3` | Topological gain weight (Step 5) |
| `--epsilon` | `0.02` | Spectral-distance (cosine) convergence threshold |
| `--max-loop-iterations` | `10` | Maximum iterations for Steps 1–5 loop |
| `--max-recovery-per-iter` | unlimited | Budget cap: max recovery operators per iteration |
| `--success-reward` | `10.0` | Reward for success branch (Step 6) |
| `--unchanged-reward` | `-1.0` | Reward for unchanged branch (Step 6) |
| `--failure-reward` | `-10.0` | Reward for failure branch (Step 6) |
| `--human-reward` | `0.0` | Reward for human policy (Step 6) |
| `-v` | off | Verbose (DEBUG) logging |

### 5.4 Programmatic Usage

**Full pipeline from Python (image + task → PPDDL):**

```python
from pyrmdp.synthesis.config import PipelineConfig
from scripts.run_pipeline import run_pipeline

cfg = PipelineConfig(
    epsilon=0.01,
    num_robot_policies=5,
    save_intermediates=True,
    output_dir="./my_experiment",
)

ppddl = run_pipeline(
    image_paths=["scene.png"],
    task_descriptions=["Pick up the red block and place it on the blue plate."],
    config=cfg,
)
```

**Using `IterativeDomainSynthesizer` directly (from a PDDL domain):**

```python
from pyppddl.ppddl.parser import load_domain
from pyrmdp.synthesis.config import PipelineConfig
from pyrmdp.synthesis.iterative_synthesizer import IterativeDomainSynthesizer

cfg = PipelineConfig.from_yaml("pipeline.yaml")
domain = load_domain("domain.pddl")

synth = IterativeDomainSynthesizer(
    domain,
    epsilon=cfg.epsilon,
    max_iterations=cfg.max_loop_iterations,
    failure_prob=cfg.failure_prob,
    scoring_config=cfg.scoring_config(),
    emission_config=cfg.emission_config(),
    output_dir=cfg.output_dir,
    save_intermediates=cfg.save_intermediates,
    enable_mutex_pruning=cfg.enable_mutex_pruning,
)

ppddl_str = synth.run()            # Steps 1–5 loop + Step 6 emission
print(synth.summary())             # {"converged": true, "iterations": 3, ...}
cfg.write_yaml("./my_experiment/pipeline.yaml")  # save for repro
```

### 5.5 FODD Visualization

```bash
python scripts/generate_add.py
python scripts/generate_markov.py
```

Both produce interactive `.html` files viewable in a browser.

### 5.6 Evolution Graph Visualization

```bash
python scripts/visualize_evolution.py pyrmdp/test_data/1/output_batch
```

Generates a single-page interactive HTML graph (vis.js) showing the full
pipeline evolution across iterations — abstract states, original
transitions, recovery operators, and provenance traces.

**Visual encoding:**

| Element | Style | Meaning |
|---------|-------|---------|
| Solid coloured node | — | Abstract state (hue = iteration) |
| Dark edge | Solid, dark grey | Original domain transition |
| Cyan dashed edge | Dashed, `#38cbcf` | Phase 1 recovery (MWCC) |
| Coloured solid edge | Solid, hue-mapped | Phase 2 recovery (reachability closure); colour = Hamming delta |
| Grey solid edge | Solid, `#6e7681` | Provenance trace (links orphan iter-2 nodes to iter-1 origins) |

**Features:**
- **Step slider** — scrub through the timeline of operators added per iteration
- **SCC timeline** — highlights which SCCs exist at each step
- **Event log** — sidebar listing each operator with `P1`/`P2` badge and section headers (`⛓ Phase 1`, `⚡ Phase 2`)
- **Provenance edges** — two mechanisms: *operator-traced* (action name → prior recovery edge) and *Jaccard fallback* (most similar prior node for fully isolated nodes)

#### Intermediate Outputs (`--save-intermediates`)

| File | Step | Content |
|------|------|--------|
| `step0_domain_generated.pddl` | 0 | PDDL domain from VLM + LLM |
| `iter{N}_step1_failures.json` | 1 | Hallucinated failure modes per action |
| `iter{N}_step2_abstract_graph.graphml` | 2 | Abstract transition graph |
| `iter{N}_step2_abstract_states.json` | 2 | All abstract states with predicate assignments |
| `iter{N}_step2_mutex_rules.json` | R5 | Mutex rules (if enabled) |
| `iter{N}_step3_condensation.json` | 3 | SCC decomposition |
| `iter{N}_step3_dag.graphml` | 3 | Condensation DAG |
| `iter{N}_step4_augmentation_bound.json` | 4 | Sources, sinks, MSCA bound |
| `iter{N}_step5_synthesized_operators.json` | 5 | Synthesized recovery operators |
| `iter{N}_step5_stats.json` | 5 | Delta minimization statistics (phase1_ops, phase2_ops, num_wccs) |
| `pipeline_summary.json` | — | Convergence diagnostics, timings, spectral distances |
| `evolution_graph.html` | — | Interactive vis.js evolution graph (generated by `visualize_evolution.py`) |

---

## 6. References

1.  **Boutilier, C., Reiter, R., & Price, B. (2001).** Symbolic dynamic programming for first-order MDPs. *IJCAI*, 1, 690-700.
2.  **Wang, C., Joshi, S., & Khardon, R. (2008).** First order decision diagrams for relational MDPs. *Journal of Artificial Intelligence Research*, 31, 431-472.
3.  **Eswaran, K. P. & Tarjan, R. E. (1976).** Augmentation problems. *SIAM Journal on Computing*, 5(4), 653-665. *(MSCA theorem)*
