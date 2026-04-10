#!/usr/bin/env python3
"""
Generate a single-page interactive HTML visualization showing the
full evolution of the abstract graph through the pipeline:

  Step 2 (original edges) → Step 5 (recovery edges added one-by-one)
  → SCC collapse animation → final irreducible graph.

Uses vis.js (CDN) for the graph and vanilla JS for the timeline slider.

Usage
-----
    python scripts/visualize_evolution.py pyrmdp/test_data/1/output/
    python scripts/visualize_evolution.py pyrmdp/test_data/1/output/ -o evolution.html
"""

from __future__ import annotations

import argparse
import json
import html as html_mod
import sys
from pathlib import Path

import networkx as nx


def load_pipeline_data(output_dir: Path) -> dict:
    """Load all relevant pipeline output files."""
    data = {}

    # Abstract graph
    ag_path = output_dir / "iter1_step2_abstract_graph.graphml"
    if ag_path.exists():
        data["graph"] = nx.read_graphml(ag_path)

    # Abstract states (for labels)
    states_path = output_dir / "iter1_step2_abstract_states.json"
    if states_path.exists():
        with open(states_path) as f:
            data["states"] = json.load(f)

    # Condensation
    cond_path = output_dir / "iter1_step3_condensation.json"
    if cond_path.exists():
        with open(cond_path) as f:
            data["condensation"] = json.load(f)

    # Step 5 operators
    ops_path = output_dir / "iter1_step5_synthesized_operators.json"
    if ops_path.exists():
        with open(ops_path) as f:
            data["operators"] = json.load(f)

    # Step 1 failures
    fail_path = output_dir / "iter1_step1_failures.json"
    if fail_path.exists():
        with open(fail_path) as f:
            data["failures"] = json.load(f)

    # Augmentation bound
    aug_path = output_dir / "iter1_step4_augmentation_bound.json"
    if aug_path.exists():
        with open(aug_path) as f:
            data["augmentation"] = json.load(f)

    # Pipeline summary
    summary_path = output_dir / "pipeline_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data["summary"] = json.load(f)

    # Robustified PPDDL (for types, predicates, operators)
    ppddl_path = output_dir / "robustified.ppddl"
    if ppddl_path.exists():
        with open(ppddl_path) as f:
            data["ppddl_text"] = f.read()

    # Origin map (VLM vs LLM vs hallucination)
    origins_path = output_dir / "step0_origins.json"
    if origins_path.exists():
        with open(origins_path) as f:
            data["origins"] = json.load(f)

    # Merge Step 1 hallucinated predicates/types from failures (fallback)
    origins = data.get("origins", {"types": {}, "predicates": {}})
    for fpath in sorted(output_dir.glob("iter*_step1_failures.json")):
        try:
            with open(fpath) as f:
                fail_data = json.load(f)
            items = fail_data if isinstance(fail_data, list) else fail_data.get("failures", [])
            for fi in items:
                for np in fi.get("new_predicates", []):
                    if np not in origins.get("predicates", {}):
                        origins.setdefault("predicates", {})[np] = "hallucination"
                nt_raw = fi.get("new_types", [])
                if isinstance(nt_raw, dict):
                    nt_raw = list(nt_raw.keys())
                for nt in nt_raw:
                    if nt not in origins.get("types", {}):
                        origins.setdefault("types", {})[nt] = "hallucination"
        except Exception:
            pass
    data["origins"] = origins

    return data


def build_state_label_map(states: list) -> dict:
    """Map state IDs to compact labels."""
    label_map = {}
    for s in states:
        sid = s["id"]
        true_preds = s.get("true", [])
        # Compact: first 3 predicates
        short = ", ".join(true_preds[:3])
        if len(true_preds) > 3:
            short += f" +{len(true_preds)-3}"
        label_map[sid] = short
    return label_map


def build_scc_map(condensation: dict) -> dict:
    """Map state_id → scc_id."""
    scc_map = {}
    for scc_id, members in condensation.get("sccs", {}).items():
        for state_id in members:
            scc_map[state_id] = int(scc_id)
    return scc_map


def _find_matching_paren(text: str, start: int) -> int:
    """Given text[start] == '(', return the index of the matching ')'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _extract_action_blocks(ppddl_text: str) -> list[str]:
    """Extract individual (:action ...) blocks from PPDDL text."""
    blocks = []
    idx = 0
    while True:
        start = ppddl_text.find("(:action", idx)
        if start == -1:
            break
        end = _find_matching_paren(ppddl_text, start)
        blocks.append(ppddl_text[start:end + 1])
        idx = end + 1
    return blocks


def _parse_atom_list(block: str) -> tuple[list[str], list[str]]:
    """Parse an (and ...) block into (positive_atoms, negative_atoms).

    Each atom is a string like 'holding ?r ?m' or 'on-surface ?m ?s'.
    Reward/cost terms are excluded.
    """
    positive: list[str] = []
    negative: list[str] = []
    inner = block.strip()

    # Strip outer parens and optional leading 'and'
    if inner.startswith("(") and inner.endswith(")"):
        content = inner[1:-1].strip()
        if content.startswith("and"):
            content = content[3:].strip()
    else:
        content = inner

    i = 0
    while i < len(content):
        if content[i] == '(':
            end = _find_matching_paren(content, i)
            atom_str = content[i + 1:end].strip()

            if atom_str.startswith("not "):
                # Negative literal: (not (pred ?args))
                neg_inner = atom_str[4:].strip()
                if neg_inner.startswith("(") and neg_inner.endswith(")"):
                    neg_atom = neg_inner[1:-1].strip()
                    name = neg_atom.split()[0] if neg_atom else ""
                    if name not in ("increase", "decrease"):
                        negative.append(neg_atom)
            else:
                name = atom_str.split()[0] if atom_str else ""
                if name not in ("and", "increase", "decrease", "probabilistic"):
                    positive.append(atom_str)

            i = end + 1
        else:
            i += 1

    return positive, negative


def _pretty_print_action(block: str) -> str:
    """Re-indent a raw (:action ...) block for readable PPDDL display.

    Strips the _human / _robot1 suffix from the action name so the
    user sees only the base operator name.  Filters out reward/cost
    ``(increase (reward) …)`` noise.
    """
    import re as _re

    # Normalise whitespace into single spaces
    flat = " ".join(block.split())
    # Strip policy suffix from the action name
    flat = _re.sub(r'\(:action\s+([\w-]+?)_(robot\d+|human)',
                   r'(:action \1', flat)
    # Remove (increase (reward) ...) terms
    flat = _re.sub(r'\(\s*increase\s+\(\s*reward\s*\)\s+[^)]*\)', '', flat)
    # Clean up any double spaces left behind
    flat = _re.sub(r'  +', ' ', flat)

    # ── Tokenize into atoms and keywords ──
    # Split the flat string into logical PDDL tokens for indented output.
    lines: list[str] = []
    INDENT = "  "

    def _fmt_section(text: str, base_depth: int) -> list[str]:
        """Format a parenthesised PDDL expression with indentation."""
        result: list[str] = []
        text = text.strip()
        if not text:
            return result

        # Simple atoms (no nested parens beyond one level) → single line
        inner_depth = 0
        for ch in text:
            if ch == '(':
                inner_depth += 1
            elif ch == ')':
                inner_depth -= 1
        # If it's a short expression, keep it on one line
        if len(text) <= 72 and text.count('\n') == 0:
            result.append(INDENT * base_depth + text)
            return result

        # Otherwise, expand: put each top-level sub-expression on its own line
        result.append(INDENT * base_depth + text[:text.index('(') + 1].rstrip()
                      if '(' in text else INDENT * base_depth + text)
        # Walk top-level children
        i = text.index('(') + 1 if '(' in text else 0
        # Find the keyword (and, probabilistic, not, etc.)
        content = text[i:].strip()
        if content.startswith(('and ', 'or ', 'probabilistic ')):
            kw_end = content.index(' ')
            kw = content[:kw_end]
            result[-1] = INDENT * base_depth + "(" + kw
            content = content[kw_end:].strip()
            # Remove trailing )
            if content.endswith(')'):
                content = content[:-1].strip()
            # Now walk each child expression
            j = 0
            while j < len(content):
                while j < len(content) and content[j] == ' ':
                    j += 1
                if j >= len(content):
                    break
                if content[j] == '(':
                    end = _find_matching_paren(content, j)
                    child = content[j:end + 1]
                    # For probabilistic: prob number comes before paren
                    result.append(INDENT * (base_depth + 1) + child)
                    j = end + 1
                elif content[j].isdigit() or content[j] in '.+-':
                    # probability number
                    num_start = j
                    while j < len(content) and content[j] not in ' (':
                        j += 1
                    num = content[num_start:j].strip()
                    while j < len(content) and content[j] == ' ':
                        j += 1
                    if j < len(content) and content[j] == '(':
                        end = _find_matching_paren(content, j)
                        child = content[j:end + 1]
                        result.append(INDENT * (base_depth + 1) + num + " " + child)
                        j = end + 1
                    else:
                        result.append(INDENT * (base_depth + 1) + num)
                else:
                    j += 1
            result.append(INDENT * base_depth + ")")
        else:
            result[-1] = INDENT * base_depth + text
        return result

    # ── Extract sections ──
    # Action name line
    m = _re.match(r'\(:action\s+([\w-]+)', flat)
    action_name = m.group(1) if m else "?"
    lines.append(f"(:action {action_name}")

    # :parameters
    pm = _re.search(r':parameters\s*\(([^)]*)\)', flat)
    if pm:
        lines.append(f"{INDENT}:parameters ({pm.group(1).strip()})")

    # :precondition
    prec_idx = flat.find(":precondition")
    if prec_idx >= 0:
        p_start = flat.find("(", prec_idx + len(":precondition"))
        if p_start >= 0:
            p_end = _find_matching_paren(flat, p_start)
            prec_text = flat[p_start:p_end + 1]
            if len(prec_text) <= 68:
                lines.append(f"{INDENT}:precondition {prec_text}")
            else:
                lines.append(f"{INDENT}:precondition")
                lines.extend(_fmt_section(prec_text, 2))

    # :effect
    eff_idx = flat.find(":effect")
    if eff_idx >= 0:
        e_start = flat.find("(", eff_idx + len(":effect"))
        if e_start >= 0:
            e_end = _find_matching_paren(flat, e_start)
            eff_text = flat[e_start:e_end + 1]
            # Clean up empty (and ) blocks left after reward removal
            eff_clean = _re.sub(r'\(\s*and\s*\)', '', eff_text).strip()
            if eff_clean and eff_clean != '()':
                if len(eff_clean) <= 68:
                    lines.append(f"{INDENT}:effect {eff_clean}")
                else:
                    lines.append(f"{INDENT}:effect")
                    lines.extend(_fmt_section(eff_clean, 2))

    lines.append(")")
    return "\n".join(lines)


def parse_ppddl_domain_info(ppddl_text: str) -> dict:
    """
    Extract types, predicates, and action signatures from the PPDDL text.
    Returns structured data for the visualization sidebar.
    """
    import re

    info: dict = {"domain_name": "", "types": [], "predicates": [], "operators": []}
    if not ppddl_text:
        return info

    # Domain name
    m = re.search(r"\(define\s+\(domain\s+([\w-]+)\)", ppddl_text)
    if m:
        info["domain_name"] = m.group(1)

    # ── Types ──
    # Extract the (:types ...) block
    types_match = re.search(r"\(:types\s+(.*?)\)", ppddl_text, re.DOTALL)
    if types_match:
        types_block = types_match.group(1).strip()
        # Parse "child1 child2 - parent" lines
        for line in types_block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if " - " in line:
                children_str, parent = line.rsplit(" - ", 1)
                children = children_str.split()
                for c in children:
                    c = c.strip()
                    if c:
                        info["types"].append({"name": c, "parent": parent.strip()})
            else:
                for tok in line.split():
                    tok = tok.strip()
                    if tok:
                        info["types"].append({"name": tok, "parent": ""})

    # ── Predicates ──
    pred_match = re.search(r"\(:predicates\s+(.*?)\)\s*(?:\(:functions|\(:action)", ppddl_text, re.DOTALL)
    if pred_match:
        pred_block = pred_match.group(1).strip()
        # Find all (pred-name ?arg - type ...) entries
        # Use balanced paren extraction
        preds = re.findall(r"\(([^()]+)\)", pred_block)
        for p_str in preds:
            p_str = p_str.strip()
            if not p_str or p_str.startswith(";"):
                continue
            tokens = p_str.split()
            pred_name = tokens[0]
            # Parse parameters
            params = []
            i = 1
            while i < len(tokens):
                if tokens[i].startswith("?"):
                    param_name = tokens[i]
                    param_type = "object"
                    if i + 1 < len(tokens) and tokens[i + 1] == "-" and i + 2 < len(tokens):
                        param_type = tokens[i + 2]
                        i += 3
                    else:
                        i += 1
                    params.append(f"{param_name}: {param_type}")
                else:
                    i += 1

            # Categorize: object-self (unary), object-object (binary), object-environment
            # Heuristic: unary non-robot predicates = self, binary = object-object,
            # predicates involving robot or surface/location = environment
            robot_params = [p for p in params if "robot" in p]
            surface_params = [p for p in params if any(t in p for t in ("surface", "location"))]
            category = "obj-obj"
            if len(params) <= 1 and not robot_params:
                category = "obj-self"
            elif robot_params or surface_params or pred_name in (
                "arm-empty", "clear", "on-surface", "on-table", "reachable",
                "at-location", "in-workspace",
            ):
                category = "obj-env"

            info["predicates"].append({
                "name": pred_name,
                "params": ", ".join(params) if params else "(none)",
                "arity": len(params),
                "category": category,
            })

    # ── Operators (full precondition + effects) ──
    # Parse _human variants for deterministic effects; fall back to _robot1.
    action_blocks = _extract_action_blocks(ppddl_text)
    seen_bases: dict[str, dict] = {}  # base_name -> operator dict

    for block in action_blocks:
      m = re.match(r'\(:action\s+([\w-]+)', block)
      if not m:
        continue
      full_name = m.group(1)
      base_name = re.sub(r"_(robot\d+|human)$", "", full_name)
      is_human = full_name.endswith("_human")

      # Extract parameters
      pm = re.search(r':parameters\s*\(([^)]*)\)', block)
      params_str = pm.group(1).strip() if pm else ""

      # Extract precondition block
      prec_idx = block.find(":precondition")
      precond_pos: list[str] = []
      precond_neg: list[str] = []
      if prec_idx >= 0:
        p_start = block.index("(", prec_idx + len(":precondition"))
        p_end = _find_matching_paren(block, p_start)
        precond_pos, precond_neg = _parse_atom_list(block[p_start:p_end + 1])

      # Extract effect block (all outcomes)
      eff_idx = block.find(":effect")
      all_outcomes = []
      if eff_idx >= 0:
        e_start = block.index("(", eff_idx + len(":effect"))
        e_end = _find_matching_paren(block, e_start)
        effect_block = block[e_start:e_end + 1]

        # Deterministic: (and ...) or any single parenthesized effect.
        stripped_effect = effect_block.strip()
        if not stripped_effect.startswith("(probabilistic"):
          add_effs, del_effs = _parse_atom_list(effect_block)
          all_outcomes.append({
            "prob": 1.0,
            "add": add_effs,
            "del": del_effs,
          })
        else:
          # Probabilistic: walk each <probability> <effect-expr> pair using balanced parens.
          content = stripped_effect[len("(probabilistic"):].strip()
          if content.endswith(")"):
            content = content[:-1].strip()

          i = 0
          while i < len(content):
            while i < len(content) and content[i].isspace():
              i += 1
            if i >= len(content):
              break

            prob_start = i
            while i < len(content) and (content[i].isdigit() or content[i] in ".+-eE"):
              i += 1
            prob_str = content[prob_start:i].strip()
            if not prob_str:
              break

            while i < len(content) and content[i].isspace():
              i += 1
            if i >= len(content) or content[i] != '(':
              break

            expr_start = i
            expr_end = _find_matching_paren(content, expr_start)
            expr_block = content[expr_start:expr_end + 1]
            add_effs, del_effs = _parse_atom_list(expr_block)
            all_outcomes.append({
              "prob": float(prob_str),
              "add": add_effs,
              "del": del_effs,
            })
            i = expr_end + 1

      # Pretty-print the raw PPDDL block with indentation
      raw_ppddl = _pretty_print_action(block)

      if base_name not in seen_bases:
        seen_bases[base_name] = {
          "name": base_name,
          "params": params_str,
          "preconditions_pos": precond_pos,
          "preconditions_neg": precond_neg,
          "outcomes": all_outcomes,
          "raw_ppddl": raw_ppddl,
        }
      elif not seen_bases[base_name].get("outcomes"):
        # Fall back to later variants only when no outcomes were captured yet.
        seen_bases[base_name]["outcomes"] = all_outcomes
        seen_bases[base_name]["raw_ppddl"] = raw_ppddl

    info["operators"] = list(seen_bases.values())
    return info


def generate_html(data: dict) -> str:
    """Generate the full evolution HTML page."""
    graph = data.get("graph", nx.DiGraph())
    states = data.get("states", [])
    condensation = data.get("condensation", {})
    operators = data.get("operators", [])
    augmentation = data.get("augmentation", {})
    failures = data.get("failures", [])
    origins = data.get("origins", {"types": {}, "predicates": {}})

    label_map = build_state_label_map(states)
    scc_map = build_scc_map(condensation)

    # ── Parse domain info from PPDDL ──
    domain_info = parse_ppddl_domain_info(data.get("ppddl_text", ""))
    types_list = domain_info["types"]
    predicates_list = domain_info["predicates"]
    operators_list = domain_info["operators"]
    domain_name = domain_info["domain_name"]

    # Categorize predicates
    preds_self = [p for p in predicates_list if p["category"] == "obj-self"]
    preds_obj = [p for p in predicates_list if p["category"] == "obj-obj"]
    preds_env = [p for p in predicates_list if p["category"] == "obj-env"]

    # Build failure info map: action_name -> explanation
    failure_map = {}
    for f in failures:
        act = f.get("action", "")
        raw = f.get("raw_response", "")
        try:
            parsed = json.loads(raw)
            failure_map[act] = parsed.get("explanation", "")[:120]
        except Exception:
            failure_map[act] = ""

    label_map = build_state_label_map(states)
    scc_map = build_scc_map(condensation)

    # ── Build nodes JSON ──
    nodes_js = []
    for node_id in graph.nodes():
        scc_id = scc_map.get(node_id, 0)
        label = label_map.get(node_id, node_id[:8])
        # Full state info for tooltip
        state_info = next((s for s in states if s["id"] == node_id), {})
        true_preds = state_info.get("true", [])
        false_preds = state_info.get("false", [])

        tooltip_lines = [
            f"<b>{node_id}</b>",
            f"SCC: {scc_id}",
            f"<br><b>True:</b> {', '.join(true_preds)}",
        ]
        if false_preds:
            tooltip_lines.append(f"<b>False:</b> {', '.join(false_preds)}")

        nodes_js.append({
            "id": node_id,
            "label": label,
            "title": "<br>".join(tooltip_lines),
            "group": scc_id,
            "scc": scc_id,
        })

    # ── Build original edges (Step 2) ──
    original_edges_js = []
    for u, v, d in graph.edges(data=True):
        action = d.get("action", "?")
        original_edges_js.append({
            "from": u,
            "to": v,
            "label": action,
            "title": f"Action: {action}",
            "action": action,
        })

    # ── Build recovery edges (Step 5), ordered by iteration ──
    recovery_edges_js = []
    for i, op in enumerate(operators):
        # Prefer stored original node IDs (correct across re-condensations).
        # Fall back to SCC mapping for legacy data.
        from_node = op.get("sink_node", "")
        to_node = op.get("source_node", "")
        if not from_node or not to_node:
            # Legacy: map SCC IDs via initial condensation (may be wrong)
            sink_scc = str(op["sink_scc"])
            source_scc = str(op["source_scc"])
            sink_members = condensation.get("sccs", {}).get(sink_scc, [])
            source_members = condensation.get("sccs", {}).get(source_scc, [])
            from_node = sink_members[0] if sink_members else f"SCC-{sink_scc}"
            to_node = source_members[0] if source_members else f"SCC-{source_scc}"

        recovery_edges_js.append({
            "from": from_node,
            "to": to_node,
            "label": op["name"],
            "title": (
                f"<b>Recovery #{i+1}:</b> {op['name']}<br>"
                f"Δ={op['delta']} | {from_node} → {to_node}<br>"
                f"Adds: {op.get('nominal_add', [])}<br>"
                f"Dels: {op.get('nominal_del', [])}"
            ),
            "delta": op["delta"],
            "iteration": i + 1,
            "sink_scc": op.get("sink_scc", -1),
            "source_scc": op.get("source_scc", -1),
        })

    # ── Action categories for legend ──
    action_names = sorted(set(e["action"] for e in original_edges_js))

    nodes_json = json.dumps(nodes_js)
    original_edges_json = json.dumps(original_edges_js)
    recovery_edges_json = json.dumps(recovery_edges_js)
    num_sccs_initial = condensation.get("num_sccs", len(graph.nodes))
    sources_initial = augmentation.get("sources", [])
    sinks_initial = augmentation.get("sinks", [])
    msca_bound = augmentation.get("bound", "?")

    # ── Build HTML fragments for domain panels ──

    # Types panel: group by parent, with origin badges
    type_origins = origins.get("types", {})
    pred_origins = origins.get("predicates", {})

    type_groups: dict = {}
    for t in types_list:
        parent = t["parent"] or "(root)"
        type_groups.setdefault(parent, []).append(t["name"])

    types_html_parts = []
    for parent, children in type_groups.items():
        tags = []
        for c in children:
            src = type_origins.get(c, "")
            badge = ""
            if src == "vlm":
                badge = '<span class="src-badge src-vlm">VLM</span>'
            elif src == "llm":
                badge = '<span class="src-badge src-llm">LLM</span>'
            elif src == "hallucination":
                badge = '<span class="src-badge src-hall">fail</span>'
            tags.append(f'<span class="tag tag-type">{c}{badge}</span>')
        children_str = ", ".join(tags)
        types_html_parts.append(
            f'<div class="type-group">'
            f'<span class="type-parent">→ {parent}</span> '
            f'{children_str}</div>'
        )
    types_html = "\n".join(types_html_parts)

    # Predicates panel: grouped by perspective, with origin badges
    def pred_row(p):
        arity_badge = f'<span class="arity-badge">{p["arity"]}</span>'
        src = pred_origins.get(p["name"], "")
        badge = ""
        if src == "vlm":
            badge = '<span class="src-badge src-vlm">VLM</span>'
        elif src == "llm":
            badge = '<span class="src-badge src-llm">LLM</span>'
        elif src == "hallucination":
            badge = '<span class="src-badge src-hall">fail</span>'
        return (
            f'<div class="pred-row">'
            f'{arity_badge}'
            f'<span class="pred-name">{p["name"]}</span>'
            f'{badge}'
            f'<span class="pred-params">({p["params"]})</span>'
            f'</div>'
        )

    preds_self_html = "\n".join(pred_row(p) for p in preds_self)
    preds_obj_html = "\n".join(pred_row(p) for p in preds_obj)
    preds_env_html = "\n".join(pred_row(p) for p in preds_env)

    # Operators panel — raw PPDDL display
    operators_html_parts = []
    recovery_names = {r["name"] for r in operators}
    for i, op in enumerate(operators_list):
      # Origin badge
      origin = "recovery" if op["name"] in recovery_names else "original"
      origin_badge = (
        '<span class="origin-badge origin-recovery">recovery</span>'
        if origin == "recovery"
        else '<span class="origin-badge origin-step0">step 0</span>'
      )

      raw = html_mod.escape(op.get("raw_ppddl", f"(:action {op['name']} …)"))

      # Failure explanation
      fail_expl = failure_map.get(op["name"], "")
      fail_html = ""
      if fail_expl:
        fail_expl_escaped = html_mod.escape(fail_expl)
        fail_html = f'<div class="op-failure">⚠ {fail_expl_escaped}</div>'

      operators_html_parts.append(
        f'<div class="op-card">'
        f'<div class="op-header">'
        f'<span class="op-name-card">{op["name"]}</span>'
        f'{origin_badge}'
        f'</div>'
        f'<pre class="ppddl-block">{raw}</pre>'
        f'{fail_html}'
        f'</div>'
      )
    operators_html = "\n".join(operators_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pyrmdp — Graph Evolution Visualization</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; }}

  .header {{
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    padding: 16px 24px;
    border-bottom: 1px solid #30363d;
    display: flex; align-items: center; gap: 16px;
  }}
  .header h1 {{ font-size: 18px; color: #58a6ff; }}
  .header .stats {{
    display: flex; gap: 20px; margin-left: auto; font-size: 13px;
  }}
  .header .stat {{ display: flex; flex-direction: column; align-items: center; }}
  .header .stat-value {{ font-size: 20px; font-weight: 700; color: #58a6ff; }}
  .header .stat-label {{ color: #8b949e; font-size: 11px; }}

  .main {{ display: flex; height: calc(100vh - 60px); }}

  #graph-container {{
    flex: 1;
    background: #0d1117;
    position: relative;
  }}

  .sidebar {{
    width: 340px;
    background: #161b22;
    border-left: 1px solid #30363d;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }}

  .panel {{
    padding: 14px 16px;
    border-bottom: 1px solid #21262d;
  }}
  .panel h3 {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8b949e;
    margin-bottom: 10px;
  }}

  /* Timeline slider */
  .timeline {{
    padding: 14px 24px;
    background: #161b22;
    border-top: 1px solid #30363d;
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  .timeline label {{ font-size: 13px; white-space: nowrap; }}
  .timeline input[type=range] {{
    flex: 1;
    accent-color: #58a6ff;
    cursor: pointer;
  }}
  .timeline .step-label {{
    min-width: 200px;
    font-size: 13px;
    font-weight: 500;
    color: #f0883e;
  }}
  .play-btn {{
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
  }}
  .play-btn:hover {{ background: #2ea043; }}
  .play-btn.paused {{ background: #da3633; }}

  /* Legend */
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
    font-size: 12px;
  }}
  .legend-swatch {{
    width: 28px;
    height: 4px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  /* Event log */
  .event-log {{
    flex: 1;
    min-height: 300px;
    flex-shrink: 0;
    overflow-y: auto;
    padding: 0 16px 14px;
  }}
  .event {{
    padding: 6px 8px;
    margin-bottom: 4px;
    border-radius: 4px;
    font-size: 12px;
    line-height: 1.4;
    border-left: 3px solid transparent;
    transition: all 0.2s;
  }}
  .event.active {{
    background: #1c2128;
    border-left-color: #58a6ff;
  }}
  .event.past {{
    opacity: 0.5;
  }}
  .event .step-num {{
    display: inline-block;
    min-width: 20px;
    font-weight: 700;
    color: #f0883e;
  }}
  .event .op-name {{
    color: #7ee787;
  }}
  .event .delta-badge {{
    display: inline-block;
    background: #30363d;
    border-radius: 10px;
    padding: 1px 7px;
    font-size: 11px;
    color: #8b949e;
    margin-left: 4px;
  }}

  /* SCC meter */
  .scc-meter {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 8px;
  }}
  .scc-bar-bg {{
    flex: 1;
    height: 8px;
    background: #21262d;
    border-radius: 4px;
    overflow: hidden;
  }}
  .scc-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
  }}
  .scc-count {{
    font-size: 18px;
    font-weight: 700;
    min-width: 30px;
    text-align: right;
  }}

  /* ── Domain info panels ── */
  .collapsible-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    user-select: none;
  }}
  .collapsible-header .chevron {{
    font-size: 10px;
    color: #8b949e;
    transition: transform 0.2s;
  }}
  .collapsible-header.open .chevron {{
    transform: rotate(90deg);
  }}
  .collapsible-body {{
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
  }}
  .collapsible-body.open {{
    max-height: 99999px;
  }}

  .type-group {{
    padding: 3px 0;
    font-size: 12px;
    line-height: 1.6;
  }}
  .type-parent {{
    color: #8b949e;
    font-weight: 500;
  }}
  .tag {{
    display: inline-block;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 11px;
    margin: 1px 2px;
  }}
  .tag-type {{
    background: #1f3a5f;
    color: #58a6ff;
  }}

  .pred-category {{
    margin-top: 8px;
  }}
  .pred-category-header {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding: 4px 0;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .pred-cat-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .pred-row {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 0 2px 14px;
    font-size: 12px;
  }}
  .arity-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #30363d;
    color: #8b949e;
    font-size: 10px;
    font-weight: 700;
    flex-shrink: 0;
  }}
  .pred-name {{
    color: #d2a8ff;
    font-weight: 500;
  }}
  .pred-params {{
    color: #8b949e;
    font-size: 11px;
  }}

  .src-badge {{
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    border-radius: 3px;
    padding: 0 4px;
    margin-left: 4px;
    vertical-align: middle;
    letter-spacing: 0.3px;
  }}
  .src-vlm {{
    background: #1a4d2e;
    color: #7ee787;
  }}
  .src-llm {{
    background: #3d1f00;
    color: #f0883e;
  }}
  .src-hall {{
    background: #4d1a1a;
    color: #f85149;
  }}

  .op-card {{
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
  }}
  .ppddl-block {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 8px 10px;
    margin: 6px 0 2px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 10.5px;
    line-height: 1.5;
    color: #c9d1d9;
    overflow-x: auto;
    white-space: pre;
    tab-size: 2;
  }}
  .op-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }}
  .op-name-card {{
    color: #7ee787;
    font-weight: 600;
  }}
  .origin-badge {{
    font-size: 10px;
    border-radius: 10px;
    padding: 1px 7px;
    font-weight: 600;
  }}
  .origin-step0 {{
    background: #1f3a5f;
    color: #58a6ff;
  }}
  .origin-recovery {{
    background: #3d1f00;
    color: #f0883e;
  }}
  .op-params {{
    color: #8b949e;
    font-size: 11px;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
  }}
  .op-section-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: #6e7681;
    margin: 4px 0 3px;
  }}
  .op-atoms {{
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    margin-bottom: 4px;
  }}
  .atom-tag {{
    display: inline-block;
    font-size: 10px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    border-radius: 3px;
    padding: 1px 5px;
    line-height: 1.4;
  }}
  .atom-pos {{
    background: #1f2937;
    color: #93c5fd;
    border: 1px solid #1e3a5f;
  }}
  .atom-neg {{
    background: #3d1a1a;
    color: #f85149;
    border: 1px solid #4d2222;
  }}
  .atom-add {{
    background: #1a3a2a;
    color: #7ee787;
    border: 1px solid #23442e;
  }}
  .atom-del {{
    background: #3d1a1a;
    color: #f85149;
    border: 1px solid #4d2222;
  }}
  .op-empty {{
    font-size: 11px;
    color: #484f58;
    font-style: italic;
  }}
  .op-failure {{
    color: #f85149;
    font-size: 11px;
    margin-top: 4px;
    padding-left: 4px;
    border-left: 2px solid #f8514944;
  }}

  .tooltip-overlay {{
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(22, 27, 34, 0.92);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    max-width: 320px;
    pointer-events: none;
    z-index: 10;
    display: none;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🔄 pyrmdp — Graph Evolution</h1>
  <div class="stats">
    <div class="stat">
      <div class="stat-value" id="stat-nodes">24</div>
      <div class="stat-label">States</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-edges">0</div>
      <div class="stat-label">Edges</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-sccs">24</div>
      <div class="stat-label">SCCs</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-sources">8</div>
      <div class="stat-label">Sources</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-sinks">16</div>
      <div class="stat-label">Sinks</div>
    </div>
  </div>
</div>

<div class="main">
  <div id="graph-container">
    <div class="tooltip-overlay" id="tooltip"></div>
  </div>

  <div class="sidebar">
    <div class="panel">
      <h3>SCC Convergence</h3>
      <div class="scc-meter">
        <div class="scc-bar-bg">
          <div class="scc-bar-fill" id="scc-bar" style="width:100%; background: #da3633;"></div>
        </div>
        <div class="scc-count" id="scc-count">{num_sccs_initial}</div>
      </div>
      <div style="font-size:11px; color:#8b949e; margin-top:6px;">
        Initial: {num_sccs_initial} SCCs → Target: 1 (irreducible)
      </div>
    </div>

    <div class="panel">
      <h3>Edge Legend</h3>
      <div class="legend-item">
        <div class="legend-swatch" style="background: #58a6ff;"></div>
        <span>Original actions (Step 2)</span>
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background: #f0883e;"></div>
        <span>Recovery — Δ=0 (no pred change)</span>
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background: #d2a8ff;"></div>
        <span>Recovery — Δ=1</span>
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background: #7ee787;"></div>
        <span>Recovery — Δ=2</span>
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background: #f778ba;"></div>
        <span>Recovery — Δ=3</span>
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background: #da3633;"></div>
        <span>Recovery — Δ≥4</span>
      </div>
      <div class="legend-item" style="margin-top:6px;">
        <div style="width:28px; display:flex; gap:2px;">
          <div style="width:6px; height:6px; border-radius:50%; background:#f0883e;"></div>
          <div style="width:6px; height:6px; border-radius:50%; background:#da3633;"></div>
        </div>
        <span>Dashed = failure branch</span>
      </div>
    </div>

    <!-- ── Object Types ── -->
    <div class="panel">
      <div class="collapsible-header open" onclick="toggleCollapse(this)">
        <h3 style="margin-bottom:0">Object Types ({len(types_list)})</h3>
        <span class="chevron">▶</span>
      </div>
      <div class="collapsible-body open">
        <div style="margin-top:8px;">
          {types_html}
        </div>
      </div>
    </div>

    <!-- ── Predicates ── -->
    <div class="panel">
      <div class="collapsible-header open" onclick="toggleCollapse(this)">
        <h3 style="margin-bottom:0">Predicates ({len(predicates_list)})</h3>
        <span class="chevron">▶</span>
      </div>
      <div class="collapsible-body open">
        <div class="pred-category">
          <div class="pred-category-header">
            <div class="pred-cat-dot" style="background:#58a6ff;"></div>
            Object-Self ({len(preds_self)})
          </div>
          {preds_self_html if preds_self_html else '<div style="padding-left:14px;font-size:11px;color:#484f58;">—</div>'}
        </div>
        <div class="pred-category">
          <div class="pred-category-header">
            <div class="pred-cat-dot" style="background:#f0883e;"></div>
            Object-Object ({len(preds_obj)})
          </div>
          {preds_obj_html if preds_obj_html else '<div style="padding-left:14px;font-size:11px;color:#484f58;">—</div>'}
        </div>
        <div class="pred-category">
          <div class="pred-category-header">
            <div class="pred-cat-dot" style="background:#7ee787;"></div>
            Object-Environment ({len(preds_env)})
          </div>
          {preds_env_html if preds_env_html else '<div style="padding-left:14px;font-size:11px;color:#484f58;">—</div>'}
        </div>
      </div>
    </div>

    <!-- ── Generated Operators ── -->
    <div class="panel">
      <div class="collapsible-header" onclick="toggleCollapse(this)">
        <h3 style="margin-bottom:0">Operators ({len(operators_list)})</h3>
        <span class="chevron">▶</span>
      </div>
      <div class="collapsible-body">
        <div style="margin-top:8px;">
          {operators_html}
        </div>
      </div>
    </div>

    <div class="panel">
      <h3>Recovery Steps ({len(operators)} total)</h3>
    </div>
    <div class="event-log" id="event-log">
      <!-- populated by JS -->
    </div>
  </div>
</div>

<div class="timeline">
  <button class="play-btn" id="play-btn" onclick="togglePlay()">▶ Play</button>
  <label>Step:</label>
  <input type="range" id="timeline-slider" min="0" max="{len(operators) + 1}" value="0">
  <div class="step-label" id="step-label">Step 2: Original graph</div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
//  Data
// ═══════════════════════════════════════════════════════════════
const nodesData = {nodes_json};
const originalEdges = {original_edges_json};
const recoveryEdges = {recovery_edges_json};
const NUM_SCCS_INITIAL = {num_sccs_initial};

// ═══════════════════════════════════════════════════════════════
//  Color scheme
// ═══════════════════════════════════════════════════════════════
const SCC_COLORS = [
  '#6baed6','#fd8d3c','#74c476','#9e9ac8','#e377c2','#bcbd22',
  '#17becf','#ff7f0e','#d62728','#1f77b4','#2ca02c','#8c564b',
  '#e7ba52','#aec7e8','#ffbb78','#98df8a','#c5b0d5','#f7b6d2',
  '#c7c7c7','#dbdb8d','#9edae5','#393b79','#637939','#8c6d31'
];

function recoveryEdgeColor(delta) {{
  if (delta === 0) return '#f0883e';
  if (delta === 1) return '#d2a8ff';
  if (delta === 2) return '#7ee787';
  if (delta === 3) return '#f778ba';
  return '#da3633';
}}

// ═══════════════════════════════════════════════════════════════
//  Build vis.js DataSets
// ═══════════════════════════════════════════════════════════════
const nodes = new vis.DataSet(nodesData.map(n => ({{
  id: n.id,
  label: n.label,
  title: n.title,
  group: n.group,
  color: {{
    background: SCC_COLORS[n.scc % SCC_COLORS.length],
    border: '#fff',
    highlight: {{ background: '#fff', border: SCC_COLORS[n.scc % SCC_COLORS.length] }}
  }},
  font: {{ color: '#c9d1d9', size: 10 }},
  shape: 'dot',
  size: 14,
  borderWidth: 1.5,
}})));

const edges = new vis.DataSet();

// Edge ID counters
let edgeId = 0;
const originalEdgeIds = [];
const recoveryEdgeIds = [];

// ═══════════════════════════════════════════════════════════════
//  Network
// ═══════════════════════════════════════════════════════════════
const container = document.getElementById('graph-container');
const network = new vis.Network(container, {{ nodes, edges }}, {{
  physics: {{
    forceAtlas2Based: {{
      gravitationalConstant: -40,
      centralGravity: 0.005,
      springLength: 140,
      springConstant: 0.06,
      damping: 0.4,
    }},
    solver: 'forceAtlas2Based',
    stabilization: {{ iterations: 300, updateInterval: 25 }},
  }},
  edges: {{
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
    smooth: {{ type: 'curvedCW', roundness: 0.15 }},
    font: {{ size: 9, color: '#8b949e', strokeWidth: 0, align: 'top' }},
  }},
  nodes: {{
    font: {{ color: '#c9d1d9', size: 10 }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    zoomView: true,
    dragView: true,
  }},
}});

// ═══════════════════════════════════════════════════════════════
//  State management
// ═══════════════════════════════════════════════════════════════
let currentStep = -1;  // -1 = nothing shown

// SCC tracking: we recalculate when recovery edges are added
// For visualization, we track how many SCCs merge at each step
// Pre-computed: simulate the SCC collapse
function computeSCCTimeline() {{
  // Build adjacency at each step
  const adj = new Map();
  nodesData.forEach(n => adj.set(n.id, new Set()));

  // Add original edges
  originalEdges.forEach(e => {{
    adj.get(e.from)?.add(e.to);
  }});

  const countSCCs = () => {{
    const visited = new Set();
    const finished = [];
    const nodeIds = nodesData.map(n => n.id);

    // DFS for Kosaraju's
    const dfs1 = (u) => {{
      visited.add(u);
      for (const v of (adj.get(u) || [])) {{
        if (!visited.has(v)) dfs1(v);
      }}
      finished.push(u);
    }};

    // Reverse graph
    const radj = new Map();
    nodeIds.forEach(n => radj.set(n, new Set()));
    for (const [u, neighbors] of adj) {{
      for (const v of neighbors) {{
        radj.get(v)?.add(u);
      }}
    }}

    const dfs2 = (u, comp) => {{
      visited.add(u);
      comp.push(u);
      for (const v of (radj.get(u) || [])) {{
        if (!visited.has(v)) dfs2(v, comp);
      }}
    }};

    nodeIds.forEach(n => {{ if (!visited.has(n)) dfs1(n); }});
    visited.clear();

    let sccs = 0;
    for (let i = finished.length - 1; i >= 0; i--) {{
      if (!visited.has(finished[i])) {{
        const comp = [];
        dfs2(finished[i], comp);
        sccs++;
      }}
    }}
    return sccs;
  }};

  const timeline = [countSCCs()]; // after original edges

  // Add recovery edges one by one
  recoveryEdges.forEach(re => {{
    adj.get(re.from)?.add(re.to);
    timeline.push(countSCCs());
  }});

  return timeline;
}}

const sccTimeline = computeSCCTimeline();

function setStep(step) {{
  // step 0 = original edges only
  // step 1..N = recovery edges added incrementally
  if (step === currentStep) return;

  const edgesToAdd = [];
  const edgeIdsToRemove = [];

  if (step < currentStep) {{
    // Going backward: rebuild from scratch
    edges.clear();
    originalEdgeIds.length = 0;
    recoveryEdgeIds.length = 0;
    edgeId = 0;
    currentStep = -1;
  }}

  // Add original edges if not yet added
  if (currentStep < 0 && step >= 0) {{
    originalEdges.forEach(e => {{
      const id = `e_${{edgeId++}}`;
      originalEdgeIds.push(id);
      edgesToAdd.push({{
        id,
        from: e.from,
        to: e.to,
        label: e.label,
        title: e.title,
        color: {{ color: '#58a6ff', highlight: '#79c0ff', opacity: 0.85 }},
        width: 2,
        dashes: false,
      }});
    }});
    currentStep = 0;
  }}

  // Add recovery edges up to `step`
  while (currentStep < step && currentStep < recoveryEdges.length) {{
    const re = recoveryEdges[currentStep]; // 0-indexed
    const id = `r_${{edgeId++}}`;
    recoveryEdgeIds.push(id);
    const c = recoveryEdgeColor(re.delta);
    edgesToAdd.push({{
      id,
      from: re.from,
      to: re.to,
      label: re.label,
      title: re.title,
      color: {{ color: c, highlight: '#fff', opacity: 0.9 }},
      width: 2.5,
      dashes: [8, 4],
    }});
    currentStep++;
  }}

  if (step > recoveryEdges.length) currentStep = recoveryEdges.length;

  if (edgesToAdd.length > 0) {{
    edges.add(edgesToAdd);
  }}

  updateUI(step);
}}

function updateUI(step) {{
  // Stats
  const totalEdges = (step >= 0 ? originalEdges.length : 0)
                   + Math.min(Math.max(step, 0), recoveryEdges.length);
  document.getElementById('stat-edges').textContent = totalEdges;

  const sccIdx = Math.min(Math.max(step, 0), sccTimeline.length - 1);
  const sccs = sccTimeline[sccIdx];
  document.getElementById('stat-sccs').textContent = sccs;
  document.getElementById('scc-count').textContent = sccs;

  const pct = (sccs / NUM_SCCS_INITIAL) * 100;
  const bar = document.getElementById('scc-bar');
  bar.style.width = pct + '%';
  bar.style.background = sccs <= 1 ? '#238636'
                        : sccs <= 5 ? '#f0883e'
                        : '#da3633';

  // Step label
  const labelEl = document.getElementById('step-label');
  if (step <= 0) {{
    labelEl.textContent = 'Step 2: Original abstract graph';
    document.getElementById('stat-sources').textContent = '{len(sources_initial)}';
    document.getElementById('stat-sinks').textContent = '{len(sinks_initial)}';
  }} else if (step <= recoveryEdges.length) {{
    const re = recoveryEdges[step - 1];
    labelEl.textContent = `Recovery #${{step}}: ${{re.label}} (Δ=${{re.delta}})`;
  }} else {{
    labelEl.textContent = '✓ Graph is irreducible (1 SCC)';
  }}

  // Event log highlighting
  document.querySelectorAll('.event').forEach((el, i) => {{
    el.classList.remove('active', 'past');
    if (i < step) el.classList.add('past');
    else if (i === step) el.classList.add('active');
  }});

  // Scroll active event into view
  const activeEl = document.querySelector('.event.active');
  if (activeEl) activeEl.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
}}

// ═══════════════════════════════════════════════════════════════
//  Event log
// ═══════════════════════════════════════════════════════════════
const logEl = document.getElementById('event-log');

// Step 0: original
const ev0 = document.createElement('div');
ev0.className = 'event';
ev0.innerHTML = `<span class="step-num">0</span> Original graph: ${{originalEdges.length}} edges, ${{nodesData.length}} states`;
ev0.onclick = () => {{ slider.value = 0; setStep(0); }};
logEl.appendChild(ev0);

// Recovery steps
recoveryEdges.forEach((re, i) => {{
  const ev = document.createElement('div');
  ev.className = 'event';
  ev.innerHTML = `<span class="step-num">${{i+1}}</span> `
    + `<span class="op-name">${{re.label}}</span>`
    + `<span class="delta-badge">Δ=${{re.delta}}</span>`
    + `<br><span style="color:#8b949e;font-size:11px;">SCC-${{re.sink_scc}} → SCC-${{re.source_scc}}</span>`;
  ev.onclick = () => {{ slider.value = i + 1; setStep(i + 1); }};
  logEl.appendChild(ev);
}});

// Final
const evF = document.createElement('div');
evF.className = 'event';
evF.innerHTML = `<span class="step-num">✓</span> <b style="color:#238636;">Irreducible</b> — 1 SCC`;
evF.onclick = () => {{ slider.value = recoveryEdges.length + 1; setStep(recoveryEdges.length + 1); }};
logEl.appendChild(evF);

// ═══════════════════════════════════════════════════════════════
//  Slider
// ═══════════════════════════════════════════════════════════════
const slider = document.getElementById('timeline-slider');
slider.addEventListener('input', () => setStep(parseInt(slider.value)));

// ═══════════════════════════════════════════════════════════════
//  Collapsible panels
// ═══════════════════════════════════════════════════════════════
function toggleCollapse(headerEl) {{
  headerEl.classList.toggle('open');
  const body = headerEl.nextElementSibling;
  if (body) body.classList.toggle('open');
}}

// ═══════════════════════════════════════════════════════════════
//  Play / Pause
// ═══════════════════════════════════════════════════════════════
let playing = false;
let playInterval = null;

function togglePlay() {{
  const btn = document.getElementById('play-btn');
  if (playing) {{
    clearInterval(playInterval);
    playing = false;
    btn.textContent = '▶ Play';
    btn.classList.remove('paused');
  }} else {{
    playing = true;
    btn.textContent = '⏸ Pause';
    btn.classList.add('paused');
    if (parseInt(slider.value) >= recoveryEdges.length + 1) {{
      slider.value = 0;
      setStep(0);
    }}
    playInterval = setInterval(() => {{
      const v = parseInt(slider.value) + 1;
      if (v > recoveryEdges.length + 1) {{
        togglePlay();
        return;
      }}
      slider.value = v;
      setStep(v);
    }}, 1200);
  }}
}}

// ═══════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════
setStep(0);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate evolution visualization of the pyrmdp pipeline"
    )
    parser.add_argument(
        "output_dir",
        help="Pipeline output directory containing GraphML, JSON files",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file path (default: <output_dir>/evolution.html)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory")
        sys.exit(1)

    data = load_pipeline_data(output_dir)
    if "graph" not in data:
        print(f"Error: no iter1_step2_abstract_graph.graphml in {output_dir}")
        sys.exit(1)

    html = generate_html(data)

    out_path = args.output or str(output_dir / "evolution.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"✓ Evolution visualization saved to: {out_path}")
    print(f"  Nodes: {len(data['graph'].nodes)}")
    print(f"  Original edges: {len(data['graph'].edges)}")
    print(f"  Recovery edges: {len(data.get('operators', []))}")


if __name__ == "__main__":
    main()
