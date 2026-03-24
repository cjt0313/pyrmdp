#!/usr/bin/env python3
import sys
import os
import networkx as nx
from pyvis.network import Network
from pyrmdp.core.logic import parse_ppddl_predicates

def generate_markov_chain(domain_path: str):
    """
    Given a PPDDL domain file, generate a simplified Abstract Markov Chain Graph.
    This simulates how abstract states (predicate combinations) connect via actions.
    """
    print(f"Parsing domain file for transitions: {domain_path}")
    predicates = parse_ppddl_predicates(domain_path)
    
    if not predicates:
        print("Error: No predicates found in domain file.")
        return

    # Create Abstract State Clusters based on key predicates
    G = nx.MultiDiGraph()
    
    # Heuristic: Create 3 Abstract Macro-States to visualize flow
    # 1. State: INITIAL / PRECONDITIONS NOT MET
    node_init = "Initial State\nPreconditions Not Met"
    G.add_node(node_init, color="#D3D3D3", title="Starting configuration")

    # 2. State: READY / PRECONDITIONS MET
    # We assume 'ready' means all predicates are roughly TRUE
    pred_str = " ∧ ".join([f"{p}(?x)" for p in predicates[:3]]) # Just first few for brevity
    node_ready = f"Action Ready\n{pred_str}\n(All Constraints Satisfied)"
    G.add_node(node_ready, color="#90EE90", title="All necessary conditions are true")

    # 3. State: GOAL / SUCCESS
    node_goal = "Result State\nAction Effect Applied\nReward +"
    G.add_node(node_goal, color="#FFD700", title="Target achievement state")


    # --- TRANSITIONS ---
    # Draw generic flow from Init -> Ready -> Goal
    
    # Transition 1: Setup Actions (Move, Prepare)
    G.add_edge(node_init, node_ready, 
               label="setup-action\n(move, open, unlock...)", 
               color="blue")

    # Self-Loop: Failures during setup
    G.add_edge(node_init, node_init, 
               label="setup-fail\n(slip, error)", 
               color="red", style="dashed")

    # Transition 2: Goal Action (Close, Pick, Sample)
    G.add_edge(node_ready, node_goal, 
               label="goal-action\n(Success High Prob)", 
               color="green")
               
    # Transition 3: Goal Action Failure (Jam)
    G.add_edge(node_ready, node_ready, 
               label="goal-action\n(Fail Low Prob)", 
               color="orange")

    # Layout for better structure
    output_name = os.path.basename(domain_path).replace(".ppddl", "_markov_chain")
    print(f"Exporting Abstract Markov Chain to '{output_name}.html'...")
    
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.save_graph(f"{output_name}.html")
    print(f"Graph generated at: {os.path.abspath(f'{output_name}.html')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_markov.py <path_to_domain.ppddl>")
        sys.exit(1)

    generate_markov_chain(sys.argv[1])
