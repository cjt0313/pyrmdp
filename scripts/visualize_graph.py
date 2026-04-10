#!/usr/bin/env python3
"""
Visualize GraphML files produced by the pyrmdp pipeline.

Usage
-----
    # Interactive matplotlib window
    python scripts/visualize_graph.py pyrmdp/test_data/1/output/iter1_step2_abstract_graph.graphml

    # Save to PNG
    python scripts/visualize_graph.py pyrmdp/test_data/1/output/iter1_step2_abstract_graph.graphml -o graph.png

    # Render all GraphML files in an output directory
    python scripts/visualize_graph.py pyrmdp/test_data/1/output/ -o graphs/

    # Export to interactive HTML (no matplotlib needed)
    python scripts/visualize_graph.py pyrmdp/test_data/1/output/iter1_step2_abstract_graph.graphml --html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx


# ════════════════════════════════════════════════════════════════════
#  Matplotlib renderer
# ════════════════════════════════════════════════════════════════════

def render_matplotlib(G: nx.DiGraph, title: str = "", save_path: str | None = None):
    """Render a NetworkX DiGraph with matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Layout
    if len(G) <= 20:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    else:
        pos = nx.kamada_kawai_layout(G)

    # Node labels — use 'label' attr if present, else node id
    node_labels = {}
    for n, data in G.nodes(data=True):
        label = data.get("label", str(n))
        # Truncate long labels
        if len(label) > 30:
            label = label[:27] + "…"
        node_labels[n] = label

    # Edge labels — use 'action' or 'prob' attr
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        parts = []
        if "action" in data:
            parts.append(str(data["action"]))
        if "prob" in data:
            parts.append(f"p={data['prob']}")
        if parts:
            edge_labels[(u, v)] = "\n".join(parts)

    # Color nodes by SCC membership if available
    node_colors = []
    cmap = plt.cm.Set3
    for n, data in G.nodes(data=True):
        scc_id = data.get("scc", None)
        if scc_id is not None:
            node_colors.append(cmap(int(scc_id) % 12))
        else:
            node_colors.append("#6baed6")

    # Draw
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#555555",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        width=1.5,
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=7, font_color="darkred",
        )

    ax.set_title(title or "Graph Visualization", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ════════════════════════════════════════════════════════════════════
#  Interactive HTML renderer (pyvis)
# ════════════════════════════════════════════════════════════════════

def render_html(G: nx.DiGraph, title: str = "", save_path: str | None = None):
    """Render a NetworkX DiGraph as an interactive HTML file using pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed. Install with: pip install pyvis")
        print("Falling back to matplotlib …")
        render_matplotlib(G, title, save_path)
        return

    net = Network(
        height="750px", width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    net.heading = title or "Graph Visualization"

    # Color palette
    colors = [
        "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8",
        "#e377c2", "#bcbd22", "#17becf", "#ff7f0e",
        "#d62728", "#1f77b4", "#2ca02c", "#8c564b",
    ]

    for n, data in G.nodes(data=True):
        label = data.get("label", str(n))
        scc_id = data.get("scc", 0)
        color = colors[int(scc_id) % len(colors)] if scc_id else colors[0]

        # Build tooltip from all node attributes
        tooltip_parts = [f"<b>{label}</b>"]
        for k, v in data.items():
            if k != "label":
                tooltip_parts.append(f"{k}: {v}")
        tooltip = "<br>".join(tooltip_parts)

        net.add_node(str(n), label=label, color=color, title=tooltip, size=25)

    for u, v, data in G.edges(data=True):
        label_parts = []
        if "action" in data:
            label_parts.append(str(data["action"]))
        if "prob" in data:
            label_parts.append(f"p={data['prob']}")
        label = " | ".join(label_parts) if label_parts else ""

        net.add_edge(str(u), str(v), label=label, title=label)

    # Physics settings for better layout
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 150,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 1}},
        "smooth": {"type": "curvedCW", "roundness": 0.2}
      }
    }
    """)

    out = save_path or "graph.html"
    if not out.endswith(".html"):
        out = out.rsplit(".", 1)[0] + ".html"
    net.save_graph(out)
    print(f"  ✓ Saved interactive HTML: {out}")


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GraphML files from the pyrmdp pipeline",
    )
    parser.add_argument(
        "input",
        help="A .graphml file or a directory containing .graphml files",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path. For a single file: image path (e.g. graph.png). "
             "For a directory: output directory for all rendered images.",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Render as interactive HTML (using pyvis) instead of static PNG",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't open matplotlib window (only useful with -o)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        graphml_files = sorted(input_path.glob("*.graphml"))
        if not graphml_files:
            print(f"No .graphml files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(graphml_files)} GraphML file(s) in {input_path}")

        out_dir = Path(args.output) if args.output else input_path
        out_dir.mkdir(parents=True, exist_ok=True)

        for gf in graphml_files:
            G = nx.read_graphml(gf)
            title = gf.stem
            ext = ".html" if args.html else ".png"
            out_file = str(out_dir / (gf.stem + ext))

            print(f"  Rendering {gf.name} …")
            if args.html:
                render_html(G, title=title, save_path=out_file)
            else:
                render_matplotlib(G, title=title, save_path=out_file)

    elif input_path.is_file() and input_path.suffix == ".graphml":
        G = nx.read_graphml(input_path)
        title = input_path.stem
        print(f"Loaded {input_path.name}: {len(G.nodes)} nodes, {len(G.edges)} edges")

        if args.html:
            out = args.output or str(input_path.with_suffix(".html"))
            render_html(G, title=title, save_path=out)
        elif args.output:
            render_matplotlib(G, title=title, save_path=args.output)
        elif not args.no_show:
            render_matplotlib(G, title=title)
    else:
        print(f"Error: {input_path} is not a .graphml file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
