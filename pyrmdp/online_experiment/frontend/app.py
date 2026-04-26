"""
Streamlit frontend for the Online Update Experiment.

Run (on the laptop) with::

    streamlit run pyrmdp/online_experiment/frontend/app.py

Set ``PYRMDP_BACKEND_URL`` to the PC address, e.g. ``http://192.168.1.5:8000``.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND = os.environ.get("PYRMDP_BACKEND_URL", "http://localhost:8000")
POLL_INTERVAL_S = 1.5

st.set_page_config(
    page_title="pyrmdp Online Experiment",
    layout="wide",
    initial_sidebar_state="collapsed",   # better default for phones
)

# Mobile-friendly tweaks: viewport tag + tighter padding on narrow screens.
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      @media (max-width: 768px) {
        .block-container { padding: 0.6rem !important; }
        h1, h2, h3 { font-size: 1.1rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.75rem; padding: 4px 8px; }
        .stButton > button { width: 100%; padding: 14px; font-size: 1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
#  HTTP helpers
# ════════════════════════════════════════════════════════════════════

def _get(path: str, **params) -> Dict[str, Any]:
    try:
        r = requests.get(f"{BACKEND}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"GET {path} failed: {exc}")
        return {}


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"POST {path} failed: {exc}")
        return {}


# ════════════════════════════════════════════════════════════════════
#  Session state bootstrap
# ════════════════════════════════════════════════════════════════════

if "failure_mode" not in st.session_state:
    st.session_state.failure_mode = False
if "history" not in st.session_state:
    st.session_state.history = []  # List[Dict[str, Any]]
if "last_poll" not in st.session_state:
    st.session_state.last_poll = 0.0


def refresh_status() -> Dict[str, Any]:
    status = _get("/get_status")
    if status:
        st.session_state.history.append({
            "t": time.time(),
            "nodes": status.get("num_nodes", 0),
            "edges": status.get("num_edges", 0),
            "delta_w": status.get("spectral_distance", 0.0),
            "tabu": len(status.get("tabu", [])),
        })
        st.session_state.last_poll = time.time()
    return status


# ════════════════════════════════════════════════════════════════════
#  Sidebar — live dashboard (always visible)
# ════════════════════════════════════════════════════════════════════

status = refresh_status()

with st.sidebar:
    st.header("📡 Live Dashboard")
    st.caption(f"Backend: `{BACKEND}`")

    eps_spec = status.get("epsilon_spectral", 0.02)
    delta_w = status.get("spectral_distance", 0.0)
    color = "🟢" if delta_w < eps_spec else "🔴"
    st.metric(
        label=f"{color} Spectral Δ_W (ε={eps_spec})",
        value=f"{delta_w:.4f}",
    )
    st.metric("Graph Nodes", status.get("num_nodes", 0))
    st.metric("Graph Edges", status.get("num_edges", 0))
    st.metric("Tabu (Severed) Edges", len(status.get("tabu", [])))
    st.metric("New Predicates Added", len(status.get("new_predicates", [])))
    st.write(f"**Resync:** `{status.get('resync_status', 'idle')}`")

    if st.button("🔁 Force MSCA Re-synthesis"):
        with st.spinner("Running MSCA …"):
            out = _post("/re_synthesize", {})
        st.json(out)

    if st.button("↻ Refresh now"):
        st.rerun()


# ════════════════════════════════════════════════════════════════════
#  Tabs
# ════════════════════════════════════════════════════════════════════

tab_human1, tab_human2, tab_dash, tab_graph = st.tabs([
    "👤 Human 1 — Teleoperator",
    "🧠 Human 2 — Baseline",
    "📊 Dashboard",
    "🕸 Graph",
])


# ──────────────────────────────────────────────────────────────────
#  Tab 1 — Human 1 (VLA simulator)
# ──────────────────────────────────────────────────────────────────
with tab_human1:
    st.subheader("Execute the requested action")
    action_name = status.get("current_action") or "(none)"
    st.markdown(
        f"### ➡️ `CURRENT ACTION: {action_name}`"
    )
    st.caption(
        "Attempt to teleoperate the robot to perform this skill, "
        "then report the outcome below."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Success", use_container_width=True,
                     disabled=action_name == "(none)"):
            out = _post("/report_success", {"action_name": action_name})
            st.success(f"Logged · next = {out.get('next_action')}")
            st.session_state.failure_mode = False
            st.rerun()
    with c2:
        if st.button("⏸ Unchanged", use_container_width=True,
                     disabled=action_name == "(none)"):
            out = _post("/report_unchanged", {"action_name": action_name})
            st.info(f"Logged · next = {out.get('next_action')}")
            st.session_state.failure_mode = False
            st.rerun()
    with c3:
        if st.button("❌ Failure", use_container_width=True,
                     disabled=action_name == "(none)"):
            st.session_state.failure_mode = True

    # ── Conditional failure diagnostic form ──
    if st.session_state.failure_mode:
        st.divider()
        st.markdown("#### 🔍 Expert Failure Diagnostic")

        # Fetch domain info once per render (populates dropdowns)
        dom = _get("/get_domain_info")
        all_types = dom.get("types", ["object"])
        all_preds = dom.get("predicate_names", [])

        ftype = st.radio(
            "Failure category",
            options=[
                ("kinematic",
                 "🚫 Kinematic Impossibility (robot can't move this way)"),
                ("known_state",
                 "📍 Failed to a Known Trap State"),
                ("new_state_from_existing",
                 "🧩 Failed to a New Abstract State (compose from existing predicates)"),
                ("new_predicate",
                 "✨ Missing Predicate — Expert defines a new one"),
                ("new_state",
                 "📝 New State via free-text / photo"),
            ],
            format_func=lambda x: x[1],
            key="ftype_radio",
        )
        ftype_value = ftype[0]

        with st.form("failure_form", clear_on_submit=False):
            payload: Dict[str, Any] = {
                "action_name": action_name,
                "failure_type": ftype_value,
            }

            # ── Branch: free-text / photo (legacy) ────────────────
            if ftype_value == "new_state":
                cap_mode = st.radio(
                    "Photo source",
                    options=["📷 Take photo (camera)", "🖼 Upload file"],
                    horizontal=True,
                    key="cap_mode",
                )
                if cap_mode.startswith("📷"):
                    img = st.camera_input("Snap the failure scene",
                                          key="cam_capture")
                else:
                    img = st.file_uploader(
                        "Upload scene photo (optional)",
                        type=["png", "jpg", "jpeg"],
                    )
                expert_text = st.text_input(
                    "Expert predicate feedback "
                    "(e.g. `broken(cup), on_floor(cup)`)"
                )
                if img is not None:
                    payload["image_b64"] = base64.b64encode(
                        img.read()).decode("ascii")
                if expert_text:
                    payload["expert_feedback"] = expert_text

            # ── Branch: compose new state from existing predicates ─
            elif ftype_value == "new_state_from_existing":
                st.caption(
                    "Pick which predicates should be TRUE and which FALSE in the "
                    "new abstract state.  Predicates not selected are ignored "
                    "(open-world)."
                )
                c1, c2 = st.columns(2)
                with c1:
                    true_sel = st.multiselect(
                        "TRUE predicates", options=all_preds,
                        key="ns_true",
                    )
                with c2:
                    false_sel = st.multiselect(
                        "FALSE predicates",
                        options=[p for p in all_preds if p not in true_sel],
                        key="ns_false",
                    )
                state_label = st.text_input(
                    "Optional label for this state",
                    placeholder="e.g. cup-broken-on-floor",
                    key="ns_label",
                )
                payload["true_predicates"] = true_sel
                payload["false_predicates"] = false_sel
                if state_label:
                    payload["state_label"] = state_label

            # ── Branch: new predicate ────────────────────────────
            elif ftype_value == "new_predicate":
                st.caption(
                    "Define a missing predicate.  After submission the "
                    "domain is extended, the abstract graph is rebuilt and "
                    "MSCA re-synthesis re-runs if Δ_W spikes."
                )
                pname = st.text_input(
                    "Predicate name", placeholder="e.g. broken",
                    key="np_name",
                )
                arity = st.number_input(
                    "Arity (number of parameters)",
                    min_value=0, max_value=6, value=1, step=1,
                    key="np_arity",
                )
                ptypes: List[str] = []
                cols = st.columns(max(1, int(arity))) if arity else []
                for i in range(int(arity)):
                    with cols[i]:
                        t = st.selectbox(
                            f"param {i} type",
                            options=all_types,
                            index=0,
                            key=f"np_type_{i}",
                        )
                        ptypes.append(t)
                payload["predicate_name"] = pname
                payload["parameter_types"] = ptypes

            submitted = st.form_submit_button(
                "📨 Send diagnostic to backend")
            if submitted:
                with st.spinner(
                    "Processing failure · may trigger graph rebuild + MSCA …"
                ):
                    out = _post("/report_failure", payload)
                st.success("Diagnostic processed — graph updated")
                st.json(out)
                st.session_state.failure_mode = False


# ──────────────────────────────────────────────────────────────────
#  Tab 2 — Human 2 (Heuristic Baseline)
# ──────────────────────────────────────────────────────────────────
with tab_human2:
    st.subheader("Heuristic Baseline — manually list recovery skills")
    st.caption(
        "You are the human expert guessing which recovery operators the "
        "robot will need.  Each skill is logged for offline comparison "
        "against the automatically-synthesised MSCA coverage."
    )

    with st.form("baseline_form", clear_on_submit=True):
        op = st.text_input("Recovery operator name",
                           placeholder="e.g. pick-up-dropped-cup")
        note = st.text_area("Rationale / precondition (optional)",
                            height=80)
        human_id = st.text_input("Your ID", value="H2")
        if st.form_submit_button("➕ Add skill"):
            if op.strip():
                out = _post("/baseline/add_skill",
                            {"operator": op.strip(),
                             "note": note.strip() or None,
                             "human_id": human_id.strip() or "H2"})
                st.success(f"Logged ({out.get('count', 0)} total)")

    skills = _get("/baseline/list").get("skills", [])
    if skills:
        st.markdown("#### Logged skills")
        st.dataframe(pd.DataFrame(skills))
    else:
        st.info("No baseline skills logged yet.")


# ──────────────────────────────────────────────────────────────────
#  Tab 3 — Dashboard
# ──────────────────────────────────────────────────────────────────
with tab_dash:
    st.subheader("Live graph topology & spectral convergence")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("t")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Wasserstein Spectral Distance Δ_W**")
            st.line_chart(df[["delta_w"]])
            st.markdown("**Graph size**")
            st.line_chart(df[["nodes", "edges"]])
        with c2:
            st.markdown("**Tabu (severed) edge count**")
            st.line_chart(df[["tabu"]])

    st.divider()
    st.markdown("#### Tabu ledger")
    tabu = status.get("tabu", [])
    if tabu:
        st.dataframe(pd.DataFrame(tabu))
    else:
        st.info("No edges severed yet.")

    st.markdown("#### Dirichlet posteriors")
    dir_dict = status.get("dirichlet", {})
    if dir_dict:
        rows = [{"action": k, **v} for k, v in dir_dict.items()]
        st.dataframe(pd.DataFrame(rows).sort_values("p_success_lcb"))

    st.markdown("#### Event log (tail)")
    events = _get("/get_event_log", tail=25).get("events", [])
    if events:
        st.dataframe(pd.DataFrame(events))


# ──────────────────────────────────────────────────────────────────
#  Tab 4 — Live abstract graph (vis.js)
# ──────────────────────────────────────────────────────────────────

# Stable color/style mapping for policies — mirrors the offline
# `visualize_evolution.py` colour scheme so figures stay consistent.
POLICY_STYLE = {
    # base operators (Step 0/2 edges)
    "base:human":   {"color": "#a371f7", "dashes": False, "icon": "🧑"},
    "base:robot1":  {"color": "#58a6ff", "dashes": False, "icon": "🤖₁"},
    "base:robot2":  {"color": "#3fb950", "dashes": False, "icon": "🤖₂"},
    "base:robot3":  {"color": "#d29922", "dashes": False, "icon": "🤖₃"},
    "base:shared":  {"color": "#8b949e", "dashes": False, "icon": "•"},
    # recovery operators (Step 5 / online MSCA edges)
    "recovery:human":  {"color": "#bc8cff", "dashes": [6, 4], "icon": "🧑↻"},
    "recovery:robot1": {"color": "#79c0ff", "dashes": [6, 4], "icon": "🤖₁↻"},
    "recovery:robot2": {"color": "#7ee787", "dashes": [6, 4], "icon": "🤖₂↻"},
    "recovery:robot3": {"color": "#e3b341", "dashes": [6, 4], "icon": "🤖₃↻"},
    "recovery:shared": {"color": "#ffa657", "dashes": [6, 4], "icon": "↻"},
}


def _style_for_policy(policy: str) -> Dict[str, Any]:
    return POLICY_STYLE.get(policy, {"color": "#8b949e",
                                     "dashes": False,
                                     "icon": "?"})


def _build_visjs_html(graph: Dict[str, Any]) -> str:
    """Render the live abstract graph as a self-contained vis.js page."""
    nodes_js: List[Dict[str, Any]] = []
    for n in graph.get("nodes", []):
        is_trap = n.get("trap")
        nodes_js.append({
            "id": n["id"],
            "label": n.get("label") or n["id"],
            "title": n.get("title", n["id"]),
            "shape": "box" if is_trap else "ellipse",
            "color": {
                "background": "#3a1a1a" if is_trap else "#161b22",
                "border":     "#f85149" if is_trap else "#30363d",
                "highlight":  {"background": "#21262d",
                               "border": "#f0883e"},
            },
            "font": {"color": "#f85149" if is_trap else "#c9d1d9",
                     "size": 12},
            "borderWidth": 3 if is_trap else 1.5,
        })

    edges_js: List[Dict[str, Any]] = []
    for e in graph.get("edges", []):
        style = _style_for_policy(e.get("policy", "base:shared"))
        prob = e.get("prob", 0.0)
        action = e.get("action", "?")
        # Edge width encodes the LCB-confidence in P(success)
        lcb = e.get("p_success_lcb")
        width = 1.0 + (lcb * 4.0 if isinstance(lcb, (int, float)) else
                       prob * 3.0)
        label = f"{style['icon']} {action}\np={prob:.2f}"
        if isinstance(lcb, (int, float)):
            label += f"\nLCB={lcb:.2f}"
        edges_js.append({
            "from": e["from"],
            "to":   e["to"],
            "label": label,
            "title": e.get("title", action),
            "arrows": "to",
            "dashes": e["severed"] or style["dashes"],
            "color": {"color": "#6e7681" if e.get("severed") else style["color"],
                      "highlight": "#f0883e"},
            "width": width,
            "font": {"color": "#8b949e", "size": 9, "align": "middle",
                     "background": "rgba(13,17,23,0.7)"},
        })

    payload = json.dumps({"nodes": nodes_js, "edges": edges_js})

    return """
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; background:#0d1117; color:#c9d1d9;
               font-family:'Segoe UI',system-ui,sans-serif; }
  #net { width: 100%; height: 70vh; min-height: 360px; border:1px solid #30363d;
         border-radius:6px; background:#0d1117; touch-action: manipulation; }
  .legend { padding:6px 10px; font-size:11px; color:#8b949e; }
  .legend span { display:inline-block; padding:2px 8px; margin:2px;
                 border-radius:3px; }
</style></head><body>
<div class="legend">
  <span style="background:#161b22;color:#a371f7">🧑 human</span>
  <span style="background:#161b22;color:#58a6ff">🤖₁ robot1</span>
  <span style="background:#161b22;color:#3fb950">🤖₂ robot2</span>
  <span style="background:#161b22;color:#d29922">🤖₃ robot3</span>
  <span style="background:#161b22;color:#ffa657">↻ recovery (dashed)</span>
  <span style="background:#3a1a1a;color:#f85149">⚠ trap state</span>
  <span style="background:#161b22;color:#6e7681">⋯ severed (tabu)</span>
</div>
<div id="net"></div>
<script>
const data = __PAYLOAD__;
const network = new vis.Network(
  document.getElementById('net'),
  { nodes: new vis.DataSet(data.nodes),
    edges: new vis.DataSet(data.edges) },
  { physics: { stabilization: { iterations: 200 },
               barnesHut: { gravitationalConstant: -8000,
                            springLength: 160 } },
    interaction: { hover: true, tooltipDelay: 100 },
    edges: { smooth: { type: 'curvedCW', roundness: 0.15 } } }
);
</script></body></html>
""".replace("__PAYLOAD__", payload)


with tab_graph:
    st.subheader("Live abstract transition graph")
    st.caption(
        "State nodes (truth-sets), operator edges coloured by policy "
        "(human vs robot1/2/3, base vs recovery), edge thickness ∝ "
        "Dirichlet-LCB of P(success).  Trap states added online appear "
        "in red boxes; severed (Tabu) edges are greyed and dashed."
    )

    graph_payload = _get("/get_graph")
    if graph_payload:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", graph_payload.get("num_nodes", 0))
        c2.metric("Edges", graph_payload.get("num_edges", 0))
        c3.metric("Tabu", len(graph_payload.get("tabu", [])))
        c4.metric("Δ_W", f"{graph_payload.get('spectral_distance', 0.0):.4f}")

        components.html(_build_visjs_html(graph_payload), height=700,
                        scrolling=False)

        # ── Operator probability table (per-policy breakdown) ──
        edges = graph_payload.get("edges", [])
        if edges:
            df = pd.DataFrame(edges)
            df = df[["action", "policy", "from", "to", "prob",
                     "p_success_mean", "p_success_lcb", "severed", "trap"]]
            df = df.sort_values(["policy", "p_success_lcb"],
                                ascending=[True, True])
            st.markdown("#### Operator edges & probabilities")
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Trap / Tabu summaries ──
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ⚠ Trap nodes")
            traps = [n for n in graph_payload["nodes"] if n.get("trap")]
            if traps:
                st.dataframe(
                    pd.DataFrame([
                        {"id": t["id"],
                         "label": t["label"],
                         "predicates": ", ".join(t.get("true_predicates", []))}
                        for t in traps
                    ]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No trap states yet — report a `new_state` failure to add one.")
        with c2:
            st.markdown("#### ⋯ Severed (Tabu) edges")
            tabu = graph_payload.get("tabu", [])
            if tabu:
                st.dataframe(pd.DataFrame(tabu),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No edges severed yet.")


# ── Auto-refresh (lightweight) ──
time.sleep(POLL_INTERVAL_S)
if time.time() - st.session_state.last_poll > POLL_INTERVAL_S:
    st.rerun()