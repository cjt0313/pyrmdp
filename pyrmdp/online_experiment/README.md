<!-- Comprehensive documentation for the Human-in-the-Loop online experiment -->
# Online Update Experiment  (pyrmdp × pyPPDDL)

A **Human-in-the-Loop Turing Test** for the neuro-symbolic domain-robustification
pipeline. One human (H1) plays the role of a noisy robot policy (the "VLA"),
another (H2) plays a heuristic baseline that manually enumerates recovery
skills. The PC backend maintains the live `pyrmdp` abstract transition graph,
Dirichlet posteriors over per-operator outcomes, a Tabu ledger of severed
(hallucinated) edges, and triggers MSCA re-synthesis whenever the graph
topology changes and the Wasserstein spectral distance
$\Delta_W = W_1(\Lambda_\text{curr}, \Lambda_\text{prev})$ exceeds
$\varepsilon_\text{spectral}$.

> **Why this matters.** The experiment isolates the performance of the
> mathematical graph engine from the VLM. When the system is fed accurate
> physical updates (either automatically or via human expert), the Dirichlet
> Process and MWCC+MSCA algorithms must flawlessly maintain topological
> irreducibility and halt the hallucination loop. This is exactly the core
> theorem of the robustification paper.

---

## 1. Architecture

```
┌──────────── Laptop / Phone (same WiFi) ─────────────┐     REST/JSON    ┌──────────── PC ─────────────┐
│ Streamlit UI  (port 8501)                           │   ← — — — — →   │ FastAPI + pyrmdp            │
│   Tab 1 — H1 teleop outcome & expert diagnostics    │                 │   GraphManager (in-memory)  │
│   Tab 2 — H2 baseline skill log                     │                 │   DirichletCounts (per op)  │
│   Tab 3 — live Δ_W / nodes / tabu dashboard         │                 │   Tabu ledger               │
│   Tab 4 — interactive vis.js graph (policy-coloured)│                 │   OnlineUpdater             │
└─────────────────────────────────────────────────────┘                 │   MSCA re-synthesis (delta) │
                                                                        └─────────────────────────────┘
```

### 1.1 File layout

```
pyrmdp/online_experiment/
├── README.md                 ← this file
├── run_backend.sh            launch uvicorn on 0.0.0.0:8000
├── run_frontend.sh           launch streamlit on 0.0.0.0:8501
├── backend/
│   ├── __init__.py
│   ├── state_manager.py      GraphManager, DirichletCounts, TabuEntry
│   ├── online_updater.py     routes outcome → Dirichlet / severing / expansion / MSCA
│   └── server.py             FastAPI endpoints
└── frontend/
    ├── __init__.py
    ├── app.py                Streamlit UI (4 tabs + sidebar)
    └── .streamlit/config.toml
```

---

## 2. Online Update Flow (backend)

```
            ┌──────────────── Human 1 clicks an outcome ────────────────┐
            │                                                           │
    ┌───── Success ──── n_success  ↑ ─────┐                             │
    ├──── Unchanged ─── n_unchanged ↑ ────┤                             │
    │                                     ▼                             ▼
    │                              update Dirichlet                cursor
    │                                                               advances
report_failure                                                         ▲
    │                                                                   │
    ├─ kinematic              ─→ force P(succ)=0, sever every edge of the op
    │                            add Tabu entries                        │
    │                            ▼                                       │
    ├─ known_state            ─→ n_failure ↑; if LCB(P) < ε_phys → sever │
    │                            ▼                                       │
    ├─ new_state_from_existing ─→ expert picks T/F predicates →          │
    │                             build AbstractState, link as trap      │
    │                                                                   │
    ├─ new_predicate          ─→ expert types name + typed arity →      │
    │                             Domain.predicates ← Predicate()        │
    │                             rebuild_graph()                        │
    │                                                                   │
    └─ new_state              ─→ parse image/expert_feedback →           │
                                  add trap node (free-form)              │
                                                                         │
          topology changed?  Δ_W = W1(Λ_curr, Λ_prev) ──► if > ε  ──► MSCA reflex
                                                                         │
                                                                         ▼
                                                              advance to next action
```

### 2.1 Dirichlet posterior

For each operator `op` we keep per-outcome pseudo-counts
$\alpha + (n_{\text{succ}}, n_{\text{unch}}, n_{\text{fail}})$, prior $\alpha = 1$.
Posterior $P_{\text{succ}} \sim \text{Beta}(a, b)$ with
$a = n_{\text{succ}}+\alpha$ and $b = n_{\text{unch}}+n_{\text{fail}}+2\alpha$.
We report the approximate lower confidence bound (LCB, $z=1.96$):

$$\mathrm{LCB}(P_{\text{succ}}) \;=\; \frac{a}{a+b} \;-\; 1.96 \sqrt{\frac{ab}{(a+b)^2(a+b+1)}}.$$

*Kinematic kill* injects $n_{\text{fail}} \mathrel{+}= 10^6$ → LCB → 0.

### 2.2 Severing (Tabu) rule

An operator's edges are moved to the Tabu ledger when
$\mathrm{LCB}(P_{\text{succ}}) < \varepsilon_\text{phys}$ (default `0.35`).
A severed operator is hidden from the action cursor.

### 2.3 Spectral reflex

On any node/edge mutation:

1. Recompute $\Lambda = \operatorname{sort}|\operatorname{eig}(M)|$ for the
   row-normalised transition matrix $M$.
2. $\Delta_W = W_1(\Lambda_\text{curr}, \Lambda_\text{prev})$.
3. If $\Delta_W > \varepsilon_\text{spectral}$ (default `0.02`), call
   `pyrmdp.synthesis.delta_minimizer.delta_minimize` to synthesise bridging
   operators; loop up to 5 rounds or until irreducible.
4. Newly synthesised operators are appended to the domain and receive their
   own `DirichletCounts` slot.

---

## 3. Failure categories (expert routing)

The Streamlit failure form exposes **five radio options**, each with a distinct
backend code-path:

| Radio label | `failure_type` | Inputs | Backend action |
|---|---|---|---|
| 🚫 Kinematic Impossibility | `kinematic` | — | Force LCB=0, sever all edges of the op |
| 📍 Known Trap State | `known_state` | — | $n_{\text{fail}}\!\uparrow$; sever if LCB<ε_phys |
| 🧩 New Abstract State (compose from existing predicates) | `new_state_from_existing` | TRUE set, FALSE set, optional label | `AbstractState(true,false)` → hashed id → attach as trap reachable from op |
| ✨ Missing Predicate — Expert defines | `new_predicate` | name, arity, typed params | Append `Predicate(name, params)` to `Domain`, `rebuild_graph()` |
| 📝 New State via free-text / photo | `new_state` | camera or upload + free-text | Parse predicates via VLM/regex → add trap node |

After any of the last three, Δ_W is recomputed and MSCA re-runs if the spike
exceeds ε.

---

## 4. REST contract

| Method | Path | Body / query | Purpose |
|---|---|---|---|
| GET  | `/`                   | — | Health check |
| GET  | `/get_status`         | — | Dashboard snapshot (current_action, num_nodes, num_edges, Δ_W, tabu, Dirichlet, event_log_size) |
| GET  | `/get_next_action`    | — | `{action_name}` (MCTS stand-in: least-sampled operator) |
| GET  | `/get_event_log`      | `tail=N` | Recent backend events |
| GET  | `/get_graph`          | — | vis.js-friendly JSON of the live abstract graph |
| GET  | `/get_domain_info`    | — | `{types, predicates, predicate_names, existing_states, actions}` |
| POST | `/report_success`     | `{action_name}` | H1 success |
| POST | `/report_unchanged`   | `{action_name}` | H1 neutral |
| POST | `/report_failure`     | see §4.1 | Routed through the updater |
| POST | `/re_synthesize`      | — | Force an MSCA round |
| POST | `/baseline/add_skill` | `{operator, note?, human_id?}` | Log H2 guess |
| GET  | `/baseline/list`      | — | Current H2 log |

### 4.1 `POST /report_failure` payload

```jsonc
{
  "action_name": "pick-up-from-surface_robot1",
  "failure_type": "kinematic | known_state | new_state | new_state_from_existing | new_predicate",

  // new_state (legacy free-text)
  "image_b64":        "<base64 PNG/JPEG>",
  "expert_feedback":  "broken(cup), on_floor(cup)",

  // new_predicate
  "predicate_name":   "broken",
  "parameter_types":  ["movable"],

  // new_state_from_existing
  "true_predicates":  ["arm-empty", "clear"],
  "false_predicates": ["graspable", "reachable"],
  "state_label":      "cup-out-of-reach"
}
```

Typical response:

```jsonc
{
  "action":          "...",
  "failure_type":    "...",
  "lcb":             0.17,
  "severed_edges":   [["S_a", "S_b"]],
  "trap_state":      "S_expert_c9cc75",
  "predicate":       { "name": "broken", "params": [{"name":"?x0","type":"movable"}] },
  "rebuild":         { "num_nodes": 16, "num_edges": 16 },
  "spectral_delta":  0.022,
  "resynthesis":     { "ok": true, "rounds": [...], "final_delta_w": 0.0 },
  "next_action":     "..."
}
```

---

## 5. Graph visualisation (Tab 4)

The 🕸 Graph tab embeds **vis.js** via `streamlit.components.v1.html` with the
same colour scheme as the offline `scripts/visualize_evolution.py`:

| Policy suffix | Colour | Style |
|---|---|---|
| `base:human`     | 🟣 `#a371f7` | solid |
| `base:robot1`    | 🔵 `#58a6ff` | solid |
| `base:robot2`    | 🟢 `#3fb950` | solid |
| `base:robot3`    | 🟠 `#d29922` | solid |
| `recovery:*`     | lighter variant | **dashed** |
| severed (Tabu)   | grey `#6e7681` | dashed |
| trap state       | red box, red border | — |

- **Edge width** $= 1 + 4\cdot\mathrm{LCB}(P_\text{succ})$ — widens as the
  posterior gains confidence.
- **Edge label** `🤖₁ pick-up…  p=0.33  LCB=0.21`.
- **Hover tooltip** shows the operator name, policy class, raw probability,
  full Dirichlet counts, LCB vs. ε_phys and Tabu status.
- Below the canvas: an **operator-edge table**, **trap-nodes table**, and
  **tabu ledger**. Legend is rendered inline at the top of the iframe.

---

## 6. Running

### 6.1 PC (backend)

```bash
cd /home/jim/code/tools/pyrmdp
pip install fastapi uvicorn pydantic streamlit requests pandas   # plus pyrmdp + pyPPDDL

export PYRMDP_DOMAIN_PATH=./pipeline_output/robustified.ppddl
export PYRMDP_EPS_PHYS=0.35
export PYRMDP_EPS_SPECTRAL=0.02
export PYRMDP_OUTPUT_DIR=./online_output

./pyrmdp/online_experiment/run_backend.sh
# → http://0.0.0.0:8000
```

### 6.2 Laptop / phone (frontend)

```bash
export PYRMDP_BACKEND_URL=http://<PC-IP>:8000          # laptop talks to PC
./pyrmdp/online_experiment/run_frontend.sh        # listens on 0.0.0.0:8501
```

Phone (same WiFi) → `http://<laptop-IP>:8501`.

> **Camera on phone.** `st.camera_input` requires either `localhost` or
> **HTTPS**. For quick WiFi demos use the **🖼 Upload file** branch (iOS/Android
> let you pick "Take Photo or Video" from the picker without HTTPS), or put
> Streamlit behind Caddy
> (`caddy reverse-proxy --from :8443 --to :8501`) for a self-signed TLS cert.

### 6.3 Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PYRMDP_DOMAIN_PATH`  | `./pipeline_output/robustified.ppddl` | PPDDL domain loaded at startup |
| `PYRMDP_EPS_PHYS`     | `0.35` | LCB threshold for severing operators |
| `PYRMDP_EPS_SPECTRAL` | `0.02` | Wasserstein convergence threshold |
| `PYRMDP_OUTPUT_DIR`   | `./online_output` | Artifacts + `baseline_skills.jsonl` |
| `PYRMDP_BACKEND_URL`  | `http://localhost:8000` | Frontend → backend URL |

---

## 7. Validation (smoke tests)

All flows below have been exercised end-to-end. Quick curl examples assuming
the backend is running at `127.0.0.1:8765`:

```bash
# 1. Kinematic kill → sever edges → Δ_W spike → MSCA synthesises recovery ops → Δ_W → 0
curl -X POST localhost:8765/report_failure -H 'Content-Type: application/json' \
  -d '{"action_name":"pick-up-from-surface_robot1","failure_type":"kinematic"}'

# 2. Expert composes a new abstract state
curl -X POST localhost:8765/report_failure -H 'Content-Type: application/json' \
  -d '{"action_name":"pick-up-from-surface_robot1","failure_type":"new_state_from_existing",
       "true_predicates":["arm-empty","clear"],
       "false_predicates":["graspable","reachable"],
       "state_label":"cup-out-of-reach"}'

# 3. Expert declares a missing predicate
curl -X POST localhost:8765/report_failure -H 'Content-Type: application/json' \
  -d '{"action_name":"pick-up-from-surface_robot2","failure_type":"new_predicate",
       "predicate_name":"broken","parameter_types":["movable"]}'

# 4. Inspect the graph
curl localhost:8765/get_graph | jq '.num_nodes,.num_edges,.spectral_distance,(.tabu|length)'
```

---

## 8. What you are proving

1. **Hallucination collapse.** Tabu ledger grows as H1 reports `kinematic`
   failures; Δ_W spikes, MSCA drives it back below ε, the graph remains
   irreducible (visible in Tab 4).
2. **Stopping-criterion efficiency.** Compare the MSCA-synthesised operator
   count against H2's manually enumerated skills
   (`online_output/baseline_skills.jsonl`). The algorithm achieves full
   physical coverage with far fewer skills and mathematically *terminates*.
3. **Expert updates preserve correctness.** The three expert routes
   (`new_predicate`, `new_state_from_existing`, `new_state`) all funnel
   through the same Dirichlet + Wasserstein reflex — identical theoretical
   guarantees regardless of whether the anomaly was discovered by a VLM or
   by a human.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: PPDDL domain not found` at startup | `PYRMDP_DOMAIN_PATH` unset / wrong | run the offline pipeline first, or set the env-var |
| Streamlit reachable only on `localhost` | bound to `127.0.0.1` | pass `--server.address 0.0.0.0` (already in `run_frontend.sh`) |
| Phone cannot reach port 8501 | Linux firewall | `sudo ufw allow 8501/tcp` |
| Camera permission denied on phone | iOS/Android require HTTPS | use **Upload file**, or proxy via Caddy |
| `/report_failure` slow on `kinematic` | MSCA re-synthesis runs in-thread | move to `BackgroundTasks` for production |
| Graph tab shows no edges after rebuild | stale edges dropped by `rebuild_graph()` | report the failure again or `POST /re_synthesize` |
