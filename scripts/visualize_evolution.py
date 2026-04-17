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
    import re as _re
    data = {}

    # ── Discover iterations ──
    iter_graph_files = sorted(output_dir.glob("iter*_step2_abstract_graph.graphml"))
    iter_nums = []
    for p in iter_graph_files:
        m = _re.search(r"iter(\d+)_", p.name)
        if m:
            iter_nums.append(int(m.group(1)))

    if not iter_nums:
        return data

    # ── Per-iteration data ──
    iterations_data: list[dict] = []
    for it in iter_nums:
        it_data: dict = {"iteration": it}

        ag_path = output_dir / f"iter{it}_step2_abstract_graph.graphml"
        if ag_path.exists():
            it_data["graph"] = nx.read_graphml(ag_path)

        states_path = output_dir / f"iter{it}_step2_abstract_states.json"
        if states_path.exists():
            with open(states_path) as f:
                it_data["states"] = json.load(f)

        cond_path = output_dir / f"iter{it}_step3_condensation.json"
        if cond_path.exists():
            with open(cond_path) as f:
                it_data["condensation"] = json.load(f)

        ops_path = output_dir / f"iter{it}_step5_synthesized_operators.json"
        if ops_path.exists():
            with open(ops_path) as f:
                it_data["operators"] = json.load(f)

        aug_path = output_dir / f"iter{it}_step4_augmentation_bound.json"
        if aug_path.exists():
            with open(aug_path) as f:
                it_data["augmentation"] = json.load(f)

        stats_path = output_dir / f"iter{it}_step5_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                it_data["step5_stats"] = json.load(f)

        iterations_data.append(it_data)

    data["iterations_data"] = iterations_data

    # Keep first iteration's data as legacy top-level keys for backward compat
    first = iterations_data[0]
    data["graph"] = first.get("graph", nx.DiGraph())
    data["states"] = first.get("states", [])
    data["condensation"] = first.get("condensation", {})
    data["operators"] = first.get("operators", [])
    data["augmentation"] = first.get("augmentation", {})

    # Step 1 failures (iter1 only for the failure_map)
    fail_path = output_dir / "iter1_step1_failures.json"
    if fail_path.exists():
        with open(fail_path) as f:
            data["failures"] = json.load(f)

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

    # LLM usage report
    usage_path = output_dir / "llm_usage.json"
    if usage_path.exists():
        with open(usage_path) as f:
            data["llm_usage"] = json.load(f)

    # Merge Step 1 hallucinated predicates/types from failures (fallback)
    # Also build per-iteration hallucination audit for the visualization.
    origins = data.get("origins", {"types": {}, "predicates": {}})
    hallucination_audit: list[dict] = []  # [{iter, base_actions, recovery_actions}]
    for fpath in sorted(output_dir.glob("iter*_step1_failures.json")):
        try:
            with open(fpath) as f:
                fail_data = json.load(f)
            items = fail_data if isinstance(fail_data, list) else fail_data.get("failures", [])
            # Extract iteration number from filename
            import re as _re
            m = _re.search(r"iter(\d+)_", fpath.name)
            iter_num = int(m.group(1)) if m else 0
            base_actions = [fi["action"] for fi in items if not fi["action"].startswith("recover_")]
            recovery_actions = [fi["action"] for fi in items if fi["action"].startswith("recover_")]
            hallucination_audit.append({
                "iteration": iter_num,
                "base_actions": base_actions,
                "recovery_actions": recovery_actions,
            })
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
    data["hallucination_audit"] = hallucination_audit

    return data


def build_state_label_map(states: list) -> dict:
    """Map state IDs to compact labels."""
    label_map = {}
    for s in states:
        sid = s["id"]
        true_preds = s.get("true", [])
        false_preds = s.get("false", [])
        parts = list(true_preds[:3])
        if len(true_preds) > 3:
            parts.append(f"+{len(true_preds)-3}")
        parts.extend(f"¬{p}" for p in false_preds[:2])
        if len(false_preds) > 2:
            parts.append(f"+{len(false_preds)-2}")
        label_map[sid] = ", ".join(parts)
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
    # Remove reward/cost annotation terms
    flat = _re.sub(r'\(\s*(?:increase|decrease)\s+\(\s*reward\s*\)\s+[^)]*\)', '', flat)
    # Clean up empty (and) blocks left after reward removal
    flat = _re.sub(r'\(\s*and\s*\)', '', flat)
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
                        # Effect was stripped (e.g. reward-only branch) → show no-op
                        result.append(INDENT * (base_depth + 1) + num + " ()")
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
    import re as _re

    graph = data.get("graph", nx.DiGraph())
    states = data.get("states", [])
    condensation = data.get("condensation", {})
    operators = data.get("operators", [])
    augmentation = data.get("augmentation", {})
    failures = data.get("failures", [])
    origins = data.get("origins", {"types": {}, "predicates": {}})
    summary = data.get("summary", {})
    hallucination_audit = data.get("hallucination_audit", [])
    iterations_data = data.get("iterations_data", [])

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

    # ── Build multi-iteration phases ──
    # Each phase: { iteration, nodes (new), originalEdges (new), recoveryEdges }
    # Nodes/edges from later iterations that don't exist in earlier ones are "new".

    # Helper functions (moved out of loop)
    def _fmt_atoms(atoms):
        parts = []
        for a in atoms:
            if isinstance(a, (list, tuple)) and len(a) > 1:
                parts.append(f"{a[0]}({', '.join(a[1:])})")
            elif isinstance(a, (list, tuple)):
                parts.append(a[0] if a else "?")
            else:
                parts.append(str(a))
        return parts

    def _fmt_params(params):
        parts = []
        for p in params:
            if isinstance(p, dict):
                parts.append(f"{p.get('name','?')} - {p.get('type','?')}")
            else:
                parts.append(str(p))
        return ", ".join(parts)

    def _build_node(node_id, it_states, it_scc_map, iteration):
        """Build a vis.js node dict for one abstract state."""
        scc_id = it_scc_map.get(node_id, 0)
        state_info = next((s for s in it_states if s["id"] == node_id), {})
        true_preds = state_info.get("true", [])
        false_preds = state_info.get("false", [])
        parts = list(true_preds[:3])
        if len(true_preds) > 3:
            parts.append(f"+{len(true_preds)-3}")
        parts.extend(f"¬{p}" for p in false_preds[:2])
        if len(false_preds) > 2:
            parts.append(f"+{len(false_preds)-2}")
        short = ", ".join(parts)
        # Full state tooltip — every predicate listed
        true_items = "".join(f"<br>&nbsp;&nbsp;✓ {p}" for p in true_preds)
        false_items = "".join(f"<br>&nbsp;&nbsp;✗ {p}" for p in false_preds)
        tooltip_lines = [
            f"<b>{node_id}</b>",
            f"SCC: {scc_id} &nbsp;|&nbsp; iter {iteration}",
            f"<br><b>True predicates ({len(true_preds)}):</b>{true_items}",
        ]
        if false_preds:
            tooltip_lines.append(
                f"<br><b>False predicates ({len(false_preds)}):</b>{false_items}"
            )
        return {
            "id": node_id,
            "label": short or node_id[:8],
            "title": "<br>".join(tooltip_lines),
            "group": scc_id,
            "scc": scc_id,
            "iteration": iteration,
        }

    all_phases = []
    seen_node_ids: set = set()
    seen_edge_keys: set = set()     # (from, to, action) tuples
    _nl_lookup: dict = {}           # cumulative recover_X -> NL description

    for it_data in iterations_data:
        it = it_data["iteration"]
        it_graph = it_data.get("graph", nx.DiGraph())
        it_states = it_data.get("states", [])
        it_cond = it_data.get("condensation", {})
        it_ops = it_data.get("operators", [])
        it_aug = it_data.get("augmentation", {})
        it_scc_map = build_scc_map(it_cond)

        # New nodes
        new_nodes = []
        for node_id in it_graph.nodes():
            if node_id not in seen_node_ids:
                new_nodes.append(_build_node(node_id, it_states, it_scc_map, it))
                seen_node_ids.add(node_id)

        # Accumulate recover_X name -> NL description across iterations
        for op in it_ops:
            oname = op.get("name", "")
            nl = op.get("nl_description", "")
            if nl:
                _nl_lookup[oname] = nl

        # New original edges
        new_orig_edges = []
        for u, v, d in it_graph.edges(data=True):
            action = d.get("action", "?")
            ekey = (u, v, action)
            if ekey not in seen_edge_keys:
                nl = _nl_lookup.get(action, "")
                label = f"{nl[:50]}\u2026\n({action})" if nl and len(nl) > 50 else f"{nl}\n({action})" if nl else action
                title = f"Action: {action}" + (f"<br><i>\U0001f4ac {nl}</i>" if nl else "")
                new_orig_edges.append({
                    "from": u, "to": v,
                    "label": label,
                    "title": title,
                    "action": action,
                })
                seen_edge_keys.add(ekey)

        # Recovery edges for this iteration
        # Determine Phase 1 vs Phase 2 split from step5 stats
        it_step5_stats = it_data.get("step5_stats", {})
        phase1_count = it_step5_stats.get("phase1_ops", 0)
        num_wccs = it_step5_stats.get("num_wccs", 1)

        it_recovery = []
        for i, op in enumerate(it_ops):
            from_node = op.get("sink_node", "")
            to_node = op.get("source_node", "")
            if not from_node or not to_node:
                sink_scc = str(op["sink_scc"])
                source_scc = str(op["source_scc"])
                sink_members = it_cond.get("sccs", {}).get(sink_scc, [])
                source_members = it_cond.get("sccs", {}).get(source_scc, [])
                from_node = sink_members[0] if sink_members else f"SCC-{sink_scc}"
                to_node = source_members[0] if source_members else f"SCC-{source_scc}"

            adds = _fmt_atoms(op.get('nominal_add', []))
            dels = _fmt_atoms(op.get('nominal_del', []))
            preconds = _fmt_atoms(op.get('precondition_atoms', []))
            params = _fmt_params(op.get('parameters', []))
            nl_desc = op.get('nl_description', '')

            # Determine edge phase from phase_tag or fallback to index-based
            phase_tag = op.get("phase_tag", "")
            if "unrolled" in phase_tag:
                edge_phase = 3
                phase_label = "Unrolled (2-hop)"
                source_badge = "🔀 Unrolled"
            elif i < phase1_count:
                edge_phase = 1
                phase_label = "MWCC"
                source_badge = "⛓ MWCC"
            else:
                edge_phase = 2
                phase_label = "Reachability"
                source_badge = "⚡ Reachability"

            # Edge label: NL sentence is primary, op name is secondary
            if nl_desc:
                short_nl = nl_desc if len(nl_desc) <= 50 else nl_desc[:47] + "…"
                edge_label = f"{short_nl}\n({op['name']})"
            else:
                edge_label = op["name"]

            # Full tooltip with all details
            pre_lines = "".join(
                f"<br>&nbsp;&nbsp;• {p}" for p in preconds
            ) if preconds else "<br>&nbsp;&nbsp;(none)"
            add_lines = "".join(
                f"<br>&nbsp;&nbsp;+ {a}" for a in adds
            ) if adds else "<br>&nbsp;&nbsp;(none)"
            del_lines = "".join(
                f"<br>&nbsp;&nbsp;− {d}" for d in dels
            ) if dels else "<br>&nbsp;&nbsp;(none)"

            tooltip = (
                f"<b>{source_badge} — {op['name']}</b><br>"
                f"Phase {edge_phase}: {phase_label} &nbsp;|&nbsp; Δ={op['delta']}<br>"
                f"{from_node} → {to_node}<br>"
            )
            if nl_desc:
                tooltip += f"<br><i>💬 {nl_desc}</i><br>"
            tooltip += (
                f"<br><b>Parameters:</b> ({params})"
                f"<br><b>Preconditions:</b>{pre_lines}"
                f"<br><b>Add effects:</b>{add_lines}"
                f"<br><b>Del effects:</b>{del_lines}"
            )

            it_recovery.append({
                "from": from_node, "to": to_node,
                "label": edge_label,
                "title": tooltip,
                "delta": op["delta"],
                "phase": edge_phase,
                "phase_tag": phase_tag,
                "nl_description": nl_desc,
                "iteration": it,
                "sink_scc": op.get("sink_scc", -1),
                "source_scc": op.get("source_scc", -1),
            })

        all_phases.append({
            "iteration": it,
            "newNodes": new_nodes,
            "newOriginalEdges": new_orig_edges,
            "recoveryEdges": it_recovery,
            "provenanceEdges": [],          # filled in post-processing below
            "numStates": len(it_graph.nodes()),
            "numEdges": len(it_graph.edges()),
            "sources": it_aug.get("sources", []),
            "sinks": it_aug.get("sinks", []),
            "numWCCs": num_wccs,
            "phase1Count": phase1_count,
        })

    # ── Post-process: add provenance edges for orphan nodes ──
    # When a later iteration has new nodes that are disconnected from
    # the old graph (e.g. precondition-states of recovery operators
    # explored in the hallucination step), add lightweight "provenance"
    # edges linking each orphan cluster back to the iter-(k-1) node
    # from which the recovery operator originated.
    for pi in range(1, len(all_phases)):
        phase = all_phases[pi]
        new_node_ids = {n["id"] for n in phase["newNodes"]}
        if not new_node_ids:
            continue

        # Collect all previously-seen node IDs (old nodes)
        old_node_ids: set = set()
        for prev_phase in all_phases[:pi]:
            for n in prev_phase["newNodes"]:
                old_node_ids.add(n["id"])

        # Find new nodes that ARE connected to old nodes via this
        # phase's original edges (in either direction)
        connected_new: set = set()
        for e in phase["newOriginalEdges"]:
            if e["from"] in old_node_ids and e["to"] in new_node_ids:
                connected_new.add(e["to"])
            if e["to"] in old_node_ids and e["from"] in new_node_ids:
                connected_new.add(e["from"])

        # Build a lookup of recovery-edge name → from-node from all
        # prior iterations
        prev_recovery_by_name: dict = {}
        for prev_phase in all_phases[:pi]:
            for re in prev_phase["recoveryEdges"]:
                prev_recovery_by_name[re["label"]] = re

        # For each new orphan source-node, trace its outgoing edge's
        # action back to a prior recovery edge and add a provenance link
        provenance: list = []
        added: set = set()
        for e in phase["newOriginalEdges"]:
            from_id = e["from"]
            if from_id not in new_node_ids:
                continue                       # old node, already connected
            if from_id in connected_new:
                continue                       # already has a link to old
            if from_id in added:
                continue                       # already handled
            action = e.get("action", "")
            if action in prev_recovery_by_name:
                prev_re = prev_recovery_by_name[action]
                provenance.append({
                    "from": prev_re["from"],   # iter-(k-1) sink-node
                    "to": from_id,
                    "label": action,
                    "title": (f"Provenance: {action}<br>"
                              f"Precondition state explored in iter {phase['iteration']}"),
                    "originAction": action,
                })
                added.add(from_id)

        # Second pass: handle completely isolated new nodes (zero edges
        # in the graph) by linking to the most-similar prior-iteration
        # node via Jaccard similarity on their true-predicate sets.
        # Build a label→id map for nodes from prior iterations.
        if pi > 0:
            # Gather all nodes that are now connected (old + connected-new + provenance targets)
            all_connected = old_node_ids | connected_new | added
            # Also mark nodes reachable from provenance targets via new edges
            # (descendants in the same orphan cluster are connected transitively)
            changed = True
            while changed:
                changed = False
                for e in phase["newOriginalEdges"]:
                    if e["from"] in all_connected and e["to"] not in all_connected:
                        all_connected.add(e["to"])
                        changed = True
                    if e["to"] in all_connected and e["from"] not in all_connected:
                        all_connected.add(e["from"])
                        changed = True

            still_isolated = new_node_ids - all_connected
            if still_isolated:
                # Build predicate-set map for old nodes
                def _extract_preds(title: str) -> frozenset:
                    """Extract true predicate names from node tooltip HTML."""
                    preds = set()
                    for pm in _re.finditer(r"\u2713 ([^<]+)", title):
                        preds.add(pm.group(1).strip())
                    return frozenset(preds)

                old_node_preds: dict = {}
                for prev_phase in all_phases[:pi]:
                    for n in prev_phase["newNodes"]:
                        old_node_preds[n["id"]] = _extract_preds(n.get("title", ""))

                # Build predicate-set map for isolated new nodes
                for iso_id in still_isolated:
                    iso_node = next((n for n in phase["newNodes"] if n["id"] == iso_id), None)
                    if not iso_node:
                        continue
                    iso_preds = _extract_preds(iso_node.get("title", ""))

                    # Find best match by Jaccard similarity
                    best_id, best_sim = None, -1.0
                    for old_id, old_preds in old_node_preds.items():
                        if not iso_preds and not old_preds:
                            sim = 0.0
                        else:
                            inter = len(iso_preds & old_preds)
                            union = len(iso_preds | old_preds)
                            sim = inter / union if union else 0.0
                        if sim > best_sim:
                            best_sim = sim
                            best_id = old_id

                    if best_id:
                        provenance.append({
                            "from": best_id,
                            "to": iso_id,
                            "label": "",
                            "title": (f"Provenance: nearest match from previous iteration (Jaccard={best_sim:.2f})<br>"
                                      f"Isolated state discovered in iter {phase['iteration']}"),
                            "originAction": "__isolated__",
                        })

        phase["provenanceEdges"] = provenance

    # Flatten for legacy code paths
    all_nodes_js = []
    for phase in all_phases:
        all_nodes_js.extend(phase["newNodes"])
    all_original_edges_js = []
    for phase in all_phases:
        all_original_edges_js.extend(phase["newOriginalEdges"])
    all_recovery_edges_js = []
    for phase in all_phases:
        all_recovery_edges_js.extend(phase["recoveryEdges"])

    phases_json = json.dumps(all_phases)
    nodes_json = json.dumps(all_nodes_js)
    original_edges_json = json.dumps(all_original_edges_js)
    recovery_edges_json = json.dumps(all_recovery_edges_js)
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
    # Build NL description lookup from all iterations' synthesized operators
    _op_nl_map: dict = {}
    for it_d in iterations_data:
        for op_d in it_d.get("operators", []):
            nl = op_d.get("nl_description", "")
            if nl:
                _op_nl_map[op_d["name"]] = nl

    for i, op in enumerate(operators_list):
      # Origin badge
      origin = "recovery" if op["name"] in recovery_names else "original"
      origin_badge = (
        '<span class="origin-badge origin-recovery">recovery</span>'
        if origin == "recovery"
        else '<span class="origin-badge origin-step0">step 0</span>'
      )

      raw = html_mod.escape(op.get("raw_ppddl", f"(:action {op['name']} …)"))

      # NL description line for recovery operators
      nl_desc = _op_nl_map.get(op["name"], "")
      nl_html = ""
      if nl_desc:
        nl_escaped = html_mod.escape(nl_desc)
        nl_html = f'<div class="op-nl-desc">💬 {nl_escaped}</div>'

      # Failure explanation
      fail_expl = failure_map.get(op["name"], "")
      fail_html = ""
      if fail_expl:
        fail_expl_escaped = html_mod.escape(fail_expl)
        fail_html = f'<div class="op-failure">⚠ {fail_expl_escaped}</div>'

      # Use NL description as display name for recovery operators
      display_name = nl_desc if nl_desc else op["name"]
      operators_html_parts.append(
        f'<div class="op-card">'
        f'<div class="op-header">'
        f'<span class="op-name-card">{html_mod.escape(display_name)}</span>'
        f'{origin_badge}'
        f'</div>'
        f'<div style="color:#8b949e;font-size:11px;padding:2px 10px;">{op["name"]}</div>'
        f'<pre class="ppddl-block">{raw}</pre>'
        f'{fail_html}'
        f'</div>'
      )
    operators_html = "\n".join(operators_html_parts)

    # ── Panel 1: Abstract-states-per-iteration bar chart (inline SVG) ──
    per_iter = summary.get("per_iteration", [])
    states_chart_html = ""
    if per_iter:
        iter_labels = [str(it["iteration"]) for it in per_iter]
        state_counts = [it["states"] for it in per_iter]
        edge_counts = [it["edges"] for it in per_iter]
        max_val = max(max(state_counts), max(edge_counts), 1)
        n = len(per_iter)
        bar_w = max(24, min(44, 260 // max(n, 1)))
        gap = 6
        svg_w = n * (bar_w + gap) + 50  # extra for y-axis label
        svg_h = 130
        chart_h = 90
        x_off = 32  # left margin for axis labels

        bars = []
        for i, (sc, ec) in enumerate(zip(state_counts, edge_counts)):
            x = x_off + i * (bar_w + gap)
            sh = max(2, sc / max_val * chart_h)
            eh = max(2, ec / max_val * chart_h)
            y_s = chart_h - sh
            y_e = chart_h - eh
            # Edges bar (behind, narrower)
            bars.append(
                f'<rect x="{x + bar_w * 0.15}" y="{y_e}" width="{bar_w * 0.7}" '
                f'height="{eh}" rx="2" fill="#30363d" opacity="0.7"/>'
            )
            # States bar (front)
            bars.append(
                f'<rect x="{x}" y="{y_s}" width="{bar_w}" height="{sh}" rx="3" '
                f'fill="#58a6ff" opacity="0.85"/>'
            )
            # Value label on top of states bar
            bars.append(
                f'<text x="{x + bar_w / 2}" y="{y_s - 3}" '
                f'text-anchor="middle" fill="#c9d1d9" font-size="10" font-weight="600">{sc}</text>'
            )
            # Iteration label
            bars.append(
                f'<text x="{x + bar_w / 2}" y="{chart_h + 14}" '
                f'text-anchor="middle" fill="#8b949e" font-size="10">It {iter_labels[i]}</text>'
            )

        # Y-axis ticks
        y_ticks = ""
        for frac in [0, 0.5, 1.0]:
            y = chart_h * (1 - frac)
            val = int(max_val * frac)
            y_ticks += (
                f'<text x="{x_off - 4}" y="{y + 3}" text-anchor="end" '
                f'fill="#484f58" font-size="9">{val}</text>'
                f'<line x1="{x_off}" y1="{y}" x2="{svg_w}" y2="{y}" '
                f'stroke="#21262d" stroke-width="0.5"/>'
            )

        states_chart_svg = (
            f'<svg width="100%" viewBox="0 0 {svg_w} {svg_h}" '
            f'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">'
            f'{y_ticks}{"".join(bars)}'
            f'</svg>'
        )
        # Legend below chart
        legend = (
            '<div style="display:flex;gap:14px;margin-top:4px;font-size:10px;color:#8b949e;">'
            '<span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
            'background:#58a6ff;margin-right:3px;"></span>States</span>'
            '<span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
            'background:#30363d;margin-right:3px;"></span>Edges</span>'
            '</div>'
        )
        converged = summary.get("converged", False)
        conv_metric = summary.get("spectral_metric", "wasserstein")
        epsilon = summary.get("epsilon", "?")
        conv_badge = (
            f'<div style="margin-top:6px;font-size:11px;">'
            f'<span style="color:{("#238636" if converged else "#da3633")};">'
            f'{"✓ Converged" if converged else "✗ Not converged"}</span>'
            f' in <b>{len(per_iter)}</b> iterations'
            f' (ε={epsilon}, {conv_metric})</div>'
        )
        states_chart_html = states_chart_svg + legend + conv_badge

    # ── Panel 2: Recovery Hallucination Audit ──
    # Frontier-Only Policy audit: each operator should be hallucinated
    # exactly once.  Violations = base ops re-appearing in iter 2+, or
    # recovery ops appearing in the same iteration as base ops that were
    # already processed.
    audit_html_parts = []
    seen_actions: set = set()
    any_violation = False
    total_base = 0
    total_recovery = 0

    if hallucination_audit:
        for entry in hallucination_audit:
            it = entry["iteration"]
            base = entry["base_actions"]
            recov = entry["recovery_actions"]
            total_base += len(base)
            total_recovery += len(recov)

            # Check for violations: any action already processed before?
            violations = [a for a in base + recov if a in seen_actions]
            if violations:
                any_violation = True
            seen_actions.update(base + recov)

            # Base actions
            base_html = ", ".join(
                f'<span class="audit-action audit-base">{html_mod.escape(a)}</span>'
                for a in base
            ) if base else '<span style="color:#484f58;font-size:11px;">—</span>'

            # Recovery actions — styled differently: expected in iter 2+
            if recov:
                recov_html = ", ".join(
                    f'<span class="audit-action audit-recovery-ok">{html_mod.escape(a)}</span>'
                    for a in recov
                )
            else:
                recov_html = '<span style="color:#484f58;font-size:11px;">—</span>'

            # Violation display
            violation_html = ""
            if violations:
                violation_html = (
                    '<div style="color:#f85149;font-size:10px;margin-top:2px;">'
                    '⚠ Re-hallucinated: ' +
                    ", ".join(f'<b>{html_mod.escape(a)}</b>' for a in violations) +
                    '</div>'
                )

            audit_html_parts.append(
                f'<div class="audit-row">'
                f'<div class="audit-iter">Iter {it}</div>'
                f'<div class="audit-detail">'
                f'<div style="margin-bottom:3px;"><b style="color:#58a6ff;">Base ({len(base)}):</b> {base_html}</div>'
                f'<div><b style="color:#f0883e;">Recovery ({len(recov)}):</b> {recov_html}</div>'
                f'{violation_html}'
                f'</div>'
                f'</div>'
            )
    else:
        audit_html_parts.append(
            '<div style="font-size:11px;color:#484f58;">No hallucination data available.</div>'
        )

    # Overall verdict
    if hallucination_audit and not any_violation:
        verdict = (
            '<div class="audit-verdict audit-verdict-ok">'
            f'✓ Frontier-Only Policy OK — {total_base} base + {total_recovery} recovery, '
            f'each hallucinated exactly once'
            '</div>'
        )
    elif any_violation:
        verdict = (
            '<div class="audit-verdict audit-verdict-bad">'
            '⚠ Frontier-Only Policy VIOLATED — operators re-hallucinated'
            '</div>'
        )
    else:
        verdict = ''
    audit_html = verdict + "\n".join(audit_html_parts)

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
  .event .phase-badge {{
    display: inline-block;
    border-radius: 10px;
    padding: 1px 7px;
    font-size: 10px;
    margin-left: 4px;
    font-weight: 600;
  }}
  .event .phase-badge.p1 {{
    background: rgba(56,203,207,0.15);
    color: #38cbcf;
    border: 1px solid rgba(56,203,207,0.3);
  }}
  .event .phase-badge.p2 {{
    background: rgba(240,136,62,0.15);
    color: #f0883e;
    border: 1px solid rgba(240,136,62,0.3);
  }}
  .event .phase-badge.p3 {{
    background: rgba(232,168,56,0.15);
    color: #e8a838;
    border: 1px solid rgba(232,168,56,0.3);
  }}
  .event-section-header {{
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
  }}
  .event-section-header.phase1 {{ color: #38cbcf; }}
  .event-section-header.phase2 {{ color: #f0883e; }}
  .event-section-header.phase3 {{ color: #e8a838; }}

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

  /* ── Iteration chart panel ── */
  .iter-chart-panel svg {{
    display: block;
    margin: 0 auto;
  }}

  /* ── Hallucination audit panel ── */
  .audit-row {{
    display: flex;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
    font-size: 12px;
  }}
  .audit-row:last-child {{ border-bottom: none; }}
  .audit-iter {{
    min-width: 42px;
    font-weight: 700;
    color: #8b949e;
    font-size: 11px;
    padding-top: 2px;
  }}
  .audit-detail {{
    flex: 1;
    line-height: 1.5;
  }}
  .audit-action {{
    display: inline-block;
    font-size: 10px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    border-radius: 3px;
    padding: 1px 5px;
    margin: 1px 2px;
  }}
  .audit-base {{
    background: #1f3a5f;
    color: #58a6ff;
  }}
  .audit-recovery-ok {{
    background: #3d2800;
    color: #f0883e;
  }}
  .audit-recovery-bad {{
    background: #4d1a1a;
    color: #f85149;
    border: 1px solid #6e2b2b;
  }}
  .audit-ok {{
    color: #238636;
    font-size: 11px;
    font-weight: 600;
  }}
  .audit-verdict {{
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .audit-verdict-ok {{
    background: #0d2818;
    color: #7ee787;
    border: 1px solid #1a4d2e;
  }}
  .audit-verdict-bad {{
    background: #3d1a1a;
    color: #f85149;
    border: 1px solid #6e2b2b;
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

    <!-- ── Abstract States per Iteration ── -->
    <div class="panel">
      <div class="collapsible-header open" onclick="toggleCollapse(this)">
        <h3 style="margin-bottom:0">Abstract States / Iteration</h3>
        <span class="chevron">▶</span>
      </div>
      <div class="collapsible-body open">
        <div class="iter-chart-panel" style="margin-top:8px;">
          {states_chart_html if states_chart_html else '<div style="font-size:11px;color:#484f58;">No iteration data.</div>'}
        </div>
      </div>
    </div>

    <!-- ── Hallucination Audit ── -->
    <div class="panel">
      <div class="collapsible-header open" onclick="toggleCollapse(this)">
        <h3 style="margin-bottom:0">Step 1 Hallucination Audit</h3>
        <span class="chevron">▶</span>
      </div>
      <div class="collapsible-body open">
        <div style="margin-top:8px;">
          {audit_html}
        </div>
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
        <span>Recovery = deterministic synthesis</span>
      </div>
      <div class="legend-item" style="margin-top:6px;">
        <div class="legend-swatch" style="background: #6e7681;"></div>
        <span>Provenance (operator origin link)</span>
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
      <h3>Recovery Steps ({len(all_recovery_edges_js)} total, {len(all_phases)} iterations)</h3>
    </div>
    <div class="event-log" id="event-log">
      <!-- populated by JS -->
    </div>
  </div>
</div>

<div class="timeline">
  <button class="play-btn" id="play-btn" onclick="togglePlay()">▶ Play</button>
  <label>Step:</label>
  <input type="range" id="timeline-slider" min="0" max="0" value="0">
  <div class="step-label" id="step-label">Iter 1: Original graph</div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
//  Data  (multi-iteration phases)
// ═══════════════════════════════════════════════════════════════
const phases       = {phases_json};
const NUM_SCCS_INI = {num_sccs_initial};

// ═══════════════════════════════════════════════════════════════
//  Color helpers
// ═══════════════════════════════════════════════════════════════
const SCC_COLORS = [
  '#6baed6','#fd8d3c','#74c476','#9e9ac8','#e377c2','#bcbd22',
  '#17becf','#ff7f0e','#d62728','#1f77b4','#2ca02c','#8c564b',
  '#e7ba52','#aec7e8','#ffbb78','#98df8a','#c5b0d5','#f7b6d2',
  '#c7c7c7','#dbdb8d','#9edae5','#393b79','#637939','#8c6d31'
];
function sccColor(idx)  {{ return SCC_COLORS[idx % SCC_COLORS.length]; }}
function recColor(delta) {{
  if (delta === 0) return '#f0883e';
  if (delta === 1) return '#d2a8ff';
  if (delta === 2) return '#7ee787';
  if (delta === 3) return '#f778ba';
  return '#da3633';
}}

// ═══════════════════════════════════════════════════════════════
//  Build flat timeline from phases
//  Each entry: {{ type, phaseIdx, ... }}
//    type='graph'    → show phase's new nodes + original edges
//    type='recovery' → add one recovery edge (recIdx into phase)
//    type='done'     → final converged marker
// ═══════════════════════════════════════════════════════════════
const timeline = [];
phases.forEach((ph, pi) => {{
  timeline.push({{ type: 'graph', phaseIdx: pi }});
  ph.recoveryEdges.forEach((_, ri) => {{
    timeline.push({{ type: 'recovery', phaseIdx: pi, recIdx: ri }});
  }});
}});
timeline.push({{ type: 'done' }});

const TOTAL_STEPS = timeline.length;   // 0 … TOTAL_STEPS-1

// ═══════════════════════════════════════════════════════════════
//  vis.js DataSets (start empty)
// ═══════════════════════════════════════════════════════════════
const nodes = new vis.DataSet();
const edges = new vis.DataSet();
let edgeId = 0;

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
  nodes: {{ font: {{ color: '#c9d1d9', size: 10 }} }},
  interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true }},
}});

// ═══════════════════════════════════════════════════════════════
//  SCC timeline (computed over the flat timeline)
// ═══════════════════════════════════════════════════════════════
function computeSCCTimeline() {{
  // Gather ALL node ids across all phases
  const allNodeIds = [];
  phases.forEach(ph => ph.newNodes.forEach(n => allNodeIds.push(n.id)));

  const adj = new Map();
  const ensureNode = (id) => {{ if (!adj.has(id)) adj.set(id, new Set()); }};

  const countSCCs = () => {{
    const nodeIds = [...adj.keys()];
    if (nodeIds.length === 0) return 0;
    const visited = new Set();
    const finished = [];
    const dfs1 = (u) => {{
      visited.add(u);
      for (const v of (adj.get(u) || [])) {{ if (!visited.has(v)) dfs1(v); }}
      finished.push(u);
    }};
    const radj = new Map();
    nodeIds.forEach(n => radj.set(n, new Set()));
    for (const [u, nbrs] of adj) {{ for (const v of nbrs) {{ radj.get(v)?.add(u); }} }}
    const dfs2 = (u, comp) => {{
      visited.add(u);
      comp.push(u);
      for (const v of (radj.get(u) || [])) {{ if (!visited.has(v)) dfs2(v, comp); }}
    }};
    nodeIds.forEach(n => {{ if (!visited.has(n)) dfs1(n); }});
    visited.clear();
    let sccs = 0;
    for (let i = finished.length - 1; i >= 0; i--) {{
      if (!visited.has(finished[i])) {{ dfs2(finished[i], []); sccs++; }}
    }}
    return sccs;
  }};

  const tl = [];
  timeline.forEach(entry => {{
    if (entry.type === 'graph') {{
      const ph = phases[entry.phaseIdx];
      ph.newNodes.forEach(n => ensureNode(n.id));
      ph.newOriginalEdges.forEach(e => {{
        ensureNode(e.from); ensureNode(e.to);
        adj.get(e.from).add(e.to);
      }});
      // Provenance edges also contribute to connectivity
      (ph.provenanceEdges || []).forEach(e => {{
        ensureNode(e.from); ensureNode(e.to);
        adj.get(e.from).add(e.to);
      }});
      tl.push(countSCCs());
    }} else if (entry.type === 'recovery') {{
      const re = phases[entry.phaseIdx].recoveryEdges[entry.recIdx];
      ensureNode(re.from); ensureNode(re.to);
      adj.get(re.from).add(re.to);
      tl.push(countSCCs());
    }} else {{
      tl.push(tl.length > 0 ? tl[tl.length - 1] : 0);
    }}
  }});
  return tl;
}}
const sccTimeline = computeSCCTimeline();

// ═══════════════════════════════════════════════════════════════
//  State: track what has been rendered up to currentStep
// ═══════════════════════════════════════════════════════════════
let currentStep = -1;

function addVisNode(n) {{
  if (nodes.get(n.id)) return;   // already present
  nodes.add({{
    id: n.id, label: n.label, title: n.title, group: n.group,
    color: {{
      background: sccColor(n.scc), border: '#fff',
      highlight: {{ background: '#fff', border: sccColor(n.scc) }}
    }},
    font: {{ color: '#c9d1d9', size: 10 }},
    shape: 'dot', size: 14, borderWidth: 1.5,
  }});
}}

function addOrigEdge(e) {{
  const id = `e_${{edgeId++}}`;
  edges.add({{
    id, from: e.from, to: e.to, label: e.label, title: e.title,
    font: {{ multi: true, size: 11, color: '#8b949e' }},
    color: {{ color: '#58a6ff', highlight: '#79c0ff', opacity: 0.85 }},
    width: 2, dashes: false,
  }});
}}

function addRecEdge(re) {{
  const id = `r_${{edgeId++}}`;
  const isP1 = re.phase === 1;
  const isP3 = re.phase === 3;
  const c = isP3 ? '#e8a838' : isP1 ? '#38cbcf' : recColor(re.delta);
  // Source badge prefix for the label
  const badge = isP3 ? '🔀 ' : isP1 ? '⛓ ' : '⚡ ';
  const lbl = badge + re.label;
  edges.add({{
    id, from: re.from, to: re.to,
    label: lbl,
    title: re.title,
    color: {{ color: c, highlight: '#fff', opacity: 0.9 }},
    font: {{ size: 11, color: '#e6edf3', strokeWidth: 2, strokeColor: '#0d1117',
             multi: true, align: 'middle' }},
    width: isP3 ? 2.5 : isP1 ? 3.0 : 2.5,
    dashes: isP3 ? [4, 4] : isP1 ? [8, 4] : false,
  }});
}}

function addProvenanceEdge(pe) {{
  const id = `p_${{edgeId++}}`;
  edges.add({{
    id, from: pe.from, to: pe.to, label: pe.label || '', title: pe.title,
    color: {{ color: '#6e7681', highlight: '#8b949e', opacity: 0.85 }},
    font: {{ size: 9, color: '#8b949e', strokeWidth: 0, align: 'top' }},
    width: 1.5,
    dashes: false,
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5, type: 'arrow' }} }},
  }});
}}

function setStep(step) {{
  step = Math.max(0, Math.min(step, TOTAL_STEPS - 1));
  if (step === currentStep) return;

  if (step < currentStep) {{
    // Going backward → rebuild from scratch
    nodes.clear(); edges.clear(); edgeId = 0; currentStep = -1;
  }}

  // Advance from currentStep+1 … step
  for (let s = currentStep + 1; s <= step; s++) {{
    const entry = timeline[s];
    if (entry.type === 'graph') {{
      const ph = phases[entry.phaseIdx];
      ph.newNodes.forEach(addVisNode);
      ph.newOriginalEdges.forEach(addOrigEdge);
      (ph.provenanceEdges || []).forEach(addProvenanceEdge);
    }} else if (entry.type === 'recovery') {{
      const re = phases[entry.phaseIdx].recoveryEdges[entry.recIdx];
      addRecEdge(re);
    }}
    // 'done' → nothing to add
  }}
  currentStep = step;
  updateUI(step);
}}

// ═══════════════════════════════════════════════════════════════
//  UI update
// ═══════════════════════════════════════════════════════════════
function updateUI(step) {{
  const entry = timeline[step];

  // Stats: nodes, edges
  document.getElementById('stat-edges').textContent = edges.length;

  const sccIdx = Math.min(step, sccTimeline.length - 1);
  const sccs = sccTimeline[sccIdx];
  document.getElementById('stat-sccs').textContent  = sccs;
  document.getElementById('scc-count').textContent   = sccs;
  const pct = (sccs / NUM_SCCS_INI) * 100;
  const bar = document.getElementById('scc-bar');
  bar.style.width = pct + '%';
  bar.style.background = sccs <= 1 ? '#238636' : sccs <= 5 ? '#f0883e' : '#da3633';

  // Sources / sinks from the current phase
  let curPhase = null;
  for (let i = step; i >= 0; i--) {{
    if (timeline[i].type === 'graph') {{ curPhase = phases[timeline[i].phaseIdx]; break; }}
  }}
  if (curPhase) {{
    document.getElementById('stat-sources').textContent = curPhase.sources.length;
    document.getElementById('stat-sinks').textContent   = curPhase.sinks.length;
  }}

  // Step label
  const labelEl = document.getElementById('step-label');
  if (entry.type === 'graph') {{
    const ph = phases[entry.phaseIdx];
    const it = ph.iteration;
    if (entry.phaseIdx === 0) {{
      labelEl.textContent = `Iter ${{it}}: Original graph (${{ph.numStates}} states, ${{ph.numEdges}} edges)`;
    }} else {{
      labelEl.textContent = `Iter ${{it}}: +${{ph.newNodes.length}} new states, +${{ph.newOriginalEdges.length}} new edges (${{ph.numStates}} total)`;
    }}
  }} else if (entry.type === 'recovery') {{
    const re = phases[entry.phaseIdx].recoveryEdges[entry.recIdx];
    const it = phases[entry.phaseIdx].iteration;
    const pLabel = re.phase === 1 ? 'Phase 1 (MWCC)' : re.phase === 3 ? 'Phase 3 (Unrolled)' : 'Phase 2 (Reachability)';
    labelEl.textContent = `Iter ${{it}} ${{pLabel}}: ${{re.label}} (Δ=${{re.delta}})`;
  }} else {{
    labelEl.textContent = '✓ Converged — graph is irreducible';
  }}

  // Event-log highlighting
  document.querySelectorAll('.event').forEach((el, i) => {{
    el.classList.remove('active', 'past');
    if (i < step) el.classList.add('past');
    else if (i === step) el.classList.add('active');
  }});
  const activeEl = document.querySelector('.event.active');
  if (activeEl) activeEl.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
}}

// ═══════════════════════════════════════════════════════════════
//  Event log  (one entry per timeline step)
// ═══════════════════════════════════════════════════════════════
const logEl  = document.getElementById('event-log');
const slider = document.getElementById('timeline-slider');

// Track when we need to insert phase section headers
let lastPhaseTag = '';

timeline.forEach((entry, idx) => {{
  // Insert section headers at Phase 1 / Phase 2 boundaries
  if (entry.type === 'recovery') {{
    const re = phases[entry.phaseIdx].recoveryEdges[entry.recIdx];
    const ph = phases[entry.phaseIdx];
    const tag = `iter${{ph.iteration}}_p${{re.phase}}`;
    if (tag !== lastPhaseTag) {{
      const hdr = document.createElement('div');
      hdr.className = 'event-section-header ' + (re.phase === 1 ? 'phase1' : re.phase === 3 ? 'phase3' : 'phase2');
      if (re.phase === 1) {{
        hdr.textContent = `⛓ Phase 1: MWCC (${{ph.numWCCs}} WCCs)`;
      }} else if (re.phase === 3) {{
        hdr.textContent = `🔀 Phase 3: Unrolled (2-hop)`;
      }} else {{
        hdr.textContent = `⚡ Phase 2: Reachability Closure`;
      }}
      logEl.appendChild(hdr);
      lastPhaseTag = tag;
    }}
  }}

  const ev = document.createElement('div');
  ev.className = 'event';

  if (entry.type === 'graph') {{
    const ph = phases[entry.phaseIdx];
    const it = ph.iteration;
    if (entry.phaseIdx === 0) {{
      ev.innerHTML = `<span class="step-num">${{idx}}</span> `
        + `<b style="color:#58a6ff;">Iter ${{it}}</b> — Original: ${{ph.numStates}} states, ${{ph.numEdges}} edges`;
    }} else {{
      ev.innerHTML = `<span class="step-num">${{idx}}</span> `
        + `<b style="color:#58a6ff;">Iter ${{it}}</b> — +${{ph.newNodes.length}} states, +${{ph.newOriginalEdges.length}} edges (${{ph.numStates}} total)`;
    }}
    lastPhaseTag = '';  // reset on new graph phase
  }} else if (entry.type === 'recovery') {{
    const re = phases[entry.phaseIdx].recoveryEdges[entry.recIdx];
    const pClass = re.phase === 1 ? 'p1' : re.phase === 3 ? 'p3' : 'p2';
    const pText  = re.phase === 1 ? 'P1' : re.phase === 3 ? 'P3' : 'P2';
    // Extract the operator name from the label (inside parentheses at the end)
    const nameMatch = re.label.match(/\(([^)]+)\)\s*$/);
    const opName = nameMatch ? nameMatch[1] : re.label.split('\\n')[0];
    // Use NL description if available, otherwise fall back to operator name
    const nlDesc = re.nl_description || '';
    const displayName = nlDesc ? nlDesc : opName;
    const shortDisplay = displayName.length > 55 ? displayName.slice(0, 52) + '…' : displayName;
    ev.innerHTML = `<span class="step-num">${{idx}}</span> `
      + `<span class="op-name">${{shortDisplay}}</span>`
      + `<span class="delta-badge">Δ=${{re.delta}}</span>`
      + `<span class="phase-badge ${{pClass}}">${{pText}}</span>`
      + `<br><span style="color:#8b949e;font-size:11px;">${{opName}} &nbsp;|&nbsp; ${{re.from}} → ${{re.to}}</span>`;
  }} else {{
    ev.innerHTML = `<span class="step-num">✓</span> <b style="color:#238636;">Converged</b> — irreducible`;
  }}

  ev.onclick = () => {{ slider.value = idx; setStep(idx); }};
  logEl.appendChild(ev);
}});

// ═══════════════════════════════════════════════════════════════
//  Slider
// ═══════════════════════════════════════════════════════════════
slider.max = TOTAL_STEPS - 1;
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
    if (parseInt(slider.value) >= TOTAL_STEPS - 1) {{
      slider.value = 0; setStep(0);
    }}
    playInterval = setInterval(() => {{
      const v = parseInt(slider.value) + 1;
      if (v >= TOTAL_STEPS) {{ togglePlay(); return; }}
      slider.value = v; setStep(v);
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

    iterations_data = data.get("iterations_data", [])
    total_nodes = sum(len(it.get("graph", nx.DiGraph()).nodes) for it in iterations_data) if iterations_data else len(data["graph"].nodes)
    total_recovery = sum(len(it.get("operators", [])) for it in iterations_data) if iterations_data else len(data.get("operators", []))
    last_graph = iterations_data[-1].get("graph", data["graph"]) if iterations_data else data["graph"]

    print(f"✓ Evolution visualization saved to: {out_path}")
    print(f"  Iterations: {len(iterations_data)}")
    print(f"  Final states: {len(last_graph.nodes)}  (iter1: {len(data['graph'].nodes)})")
    print(f"  Final edges:  {len(last_graph.edges)}")
    print(f"  Recovery ops: {total_recovery}")


if __name__ == "__main__":
    main()
