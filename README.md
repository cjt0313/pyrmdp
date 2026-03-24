# pyrmdp: Relational MDPs with First-Order Decision Diagrams (FODDs)

`pyrmdp` is a Python library designed to efficiently represent and solve **Relational Markov Decision Processes (RMDPs)** using **First-Order Decision Diagrams (FODDs)**. 
Unlike traditional MDP solvers that require **grounding** (enumerating all possible states, resulting in exponential state space explosion), `pyrmdp` works directly with the **lifted** (relational) representation found in PPDDL files. It uses logical reductions and automated reasoning to prune the state space, keeping the representation compact.

## Table of Contents

- [1. Core Concept & Theory](#1-core-concept--theory)
  - [1.1 Relational MDPs vs. Grounded MDPs](#11-relational-mdps-vs-grounded-mdps)
  - [1.2 FODDs and State Space Reduction](#12-fodds-and-state-space-reduction)
  - [1.3 The 5 Reduction Rules (R1-R5)](#13-the-5-reduction-rules-r1-r5)
- [2. Architecture & Design](#2-architecture--design)
  - [2.1 Project Structure](#21-project-structure)
  - [2.2 Logic & PPDDL Parsing (`core.logic`)](#22-logic--ppddl-parsing-corelogic)
  - [2.3 FODD Core & Manager (`core.fodd`)](#23-fodd-core--manager-corefodd)
  - [2.4 Reduction Engine (`pruning.reduction`)](#24-reduction-engine-pruningreduction)
  - [2.5 Automated Reasoning with LLMs & Z3 (`pruning.llm_axiom`)](#25-automated-reasoning-with-llms--z3-pruningllm_axiom)
- [3. Usage & Examples](#3-usage--examples)
  - [3.1 Installation](#31-installation)
  - [3.2 Running the Scripts](#32-running-the-scripts)
  - [3.3 Visualization](#33-visualization)

---

## 1. Core Concept & Theory

### 1.1 Relational MDPs vs. Grounded MDPs

*   **Grounded MDPs (Classic):** States are distinct identifiers (s1, s2, ...). A problem with 10 blocks can have millions of states. Solvers struggle with the "Curse of Dimensionality."
*   **Relational MDPs (RMDPs):** States are described by properties and relations (e.g., `on(BlockA, BlockB)`). Policies allow generalization (e.g., "If *any* block is Clear, pick it up").

### 1.2 FODDs and State Space Reduction

A **First-Order Decision Diagram (FODD)** is a data structure similar to a Binary Decision Diagram (BDD) but for First-Order Logic. It represents a value function or policy as a directed acyclic graph (DAG). 

`pyrmdp` implements techniques from the **RelationalMDPADD** paper to perform **symbolic dynamic programming**. It computes the standard **Value Function** (satisfying the Bellman equation), but instead of a tabular representation, it **automatically** discovers and groups states that share the same value, representing the function compactly using logical decision diagrams.

### 1.3 The 5 Reduction Rules (R1-R5)

To prevent the FODD graph from exploding in size, `pyrmdp` applies five reduction operators:

1.  **R1 (Neglect):** If the `True` branch and `False` branch of a node point to the same child, the node offers no information. **Delete it.**
2.  **R2 (Join):** If two nodes represent the exact same logic (same test, same children), they are identical. **Merge them.** (Implemented via the `Unique Table` in `FODDManager`).
3.  **R3 (Merge):** If a child node tests the same predicate as its parent, it is redundant. **Simplify the path.**
4.  **R4 (Sort):** Enforces a strict Global Order on predicates (e.g., `on` must always be tested before `clear`). This ensures a canonical graph structure, making R2 efficient.
5.  **R5 (Strong Reduction):** Uses background knowledge to prune impossible paths. 
    *   *Example:* If we test `on(A, B) = True`, we know `clear(B)` must be `False`. A branch testing `clear(B) = True` after `on(A, B)` is **impossible (UNSAT)** and can be pruned.
    *   `pyrmdp` uses **Z3 Theorem Prover** to verify these logical consistencies.

---

## 2. Architecture & Design

### 2.1 Project Structure

```
pyrmdp/
├── pyrmdp/
│   ├── core/              # Core data structures
│   │   ├── logic.py       # Atoms, Variables, PPDDL Parsing
│   │   └── fodd.py        # FODDNode, FODDManager (R2, R4)
│   ├── pruning/           # Reduction algorithms
│   │   ├── reduction.py   # Syntactic (R1, R3) & Strong (R5) Reducers
│   │   └── llm_axiom.py   # LLM generation of Z3 background knowledge
│   └── vis/               # Visualization tools
│       └── visualization.py # NetworkX/PyVis interactive plotting
├── scripts/               # Example usage scripts
└── setup.py               # Installation script
```

### 2.2 Logic & PPDDL Parsing (`core.logic`)

*   **Classes:** `Variable`, `Constant`, `Atom`.
*   **PPDDL Parsing:** Includes a regex-based parser (`parse_ppddl_predicates`) to extract schema information directly from `.ppddl` domain files.

### 2.3 FODD Core & Manager (`core.fodd`)

*   **FODDManager:** The central "brain" of the system.
    *   **Unique Table:** Implements **R2**. Ensures that if you request a node that already exists, you get the existing ID instead of creating a duplicate.
    *   **Global Order:** Implements **R4**. Maintains the priority list of predicates to keep the graph sorted.

### 2.4 Reduction Engine (`pruning.reduction`)

*   **SyntacticReducer:** Handles local graph simplifications (R1 and R3) that don't require external solvers.
*   **StrongReducer:** Interfaces with the **Z3 Solver**. It builds a logical path constraint for every branch. If `Constraint(path) AND BackgroundKnowledge` is unsatisfiable, the branch is pruned.

### 2.5 Automated Reasoning with LLMs & Z3 (`pruning.llm_axiom`)

A unique feature of `pyrmdp` is the ability to **automatically generate background knowledge**.
1.  The system reads predicates from your PPDDL file.
2.  It prompts an LLM (OpenAI/Azure) to intuit physical constraints (e.g., "An object cannot be at two places at once").
3.  These constraints are converted into **Z3 Axioms** used by the `StrongReducer`.

---

## 3. Usage & Examples

### 3.1 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/pyrmdp.git
cd pyrmdp

# Install dependencies and the package
pip install -e .
```

### 3.2 Running the Scripts

The `scripts/` directory contains tools to generate diagrams and visualizations.

**Generate an Algebraic Decision Diagram (ADD) from PPDDL:**

```bash
# Ensure you are in the root directory (pyrmdp)
python3 scripts/generate_add.py
```

This script parses a PPDDL domain file and generates a representative First-Order Decision Diagram (FODD) visualizing the logical structure.

**Generate a Markov Chain visualization:**

```bash
python3 scripts/generate_markov.py
```

### 3.3 Visualization

The library outputs interactive HTML files (using `pyvis`) that allow you to explore the state space structure.

*   **Nodes:** Represent logical tests (e.g., `at(?r, ?l1)`).
*   **Edges:** Green = True branch, Red = False branch.
*   **Leaves:** Represent value function values or policy actions.

After running the scripts, look for generated `.html` files in your workspace (e.g., `fodd_structure.html`).

---

## 4. References

1.  **Boutilier, C., Reiter, R., & Price, B. (2001).** Symbolic dynamic programming for first-order MDPs. *IJCAI*, 1, 690-700.
2.  **Wang, C., Joshi, S., & Khardon, R. (2008).** First order decision diagrams for relational MDPs. *Journal of Artificial Intelligence Research*, 31, 431-472.
