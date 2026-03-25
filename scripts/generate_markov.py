#!/usr/bin/env python3
import sys
import os
import argparse
import networkx as nx
from pyvis.network import Network
from pyrmdp.core.logic import parse_ppddl_actions
from pyrmdp.core.markov import AbstractTransitionMatrix, state_label



def generate_markov_chain(domain_path: str, verbosity: int = 1):
    """
    Given a PPDDL domain file, generate an Abstract Markov Chain Graph.
    This simulates how abstract states (predicate combinations) connect via actions.
    verbosity: Controls amount of information displayed in the graph.
               0: basic states and connectivity
               1 (or >0): detailed state information, predicates, edge descriptions
    """
    print(f"Parsing domain file for actions: {domain_path}")
    actions = parse_ppddl_actions(domain_path)
    
    if not actions:
        print("Error: No actions found in domain file. Make sure the file contains :action blocks.")
        return

    states_set = set()
    transitions = []
    
    G = nx.MultiDiGraph()
    state_to_id = {}
    
    def get_state_id(s: frozenset) -> str:
        if s not in state_to_id:
            state_to_id[s] = f"S{len(state_to_id)}"
        return state_to_id[s]
        
    for action in actions:
        pre_set = frozenset(action.preconditions)
        states_set.add(pre_set)
        
        for eff in action.effects:
            prob = eff["prob"]
            post_set = set(pre_set)
            for d in eff["del"]:
                post_set.discard(d)
            for a in eff["add"]:
                post_set.add(a)
            post_set = frozenset(post_set)
            states_set.add(post_set)
            
            # Record transition
            transitions.append((pre_set, post_set, prob, action.name))
            
    # Add nodes to Graph
    states_list = list(states_set)
    for s in states_list:
        s_id = get_state_id(s)
        label = state_label(s, verbosity)
        color = "#90EE90" if len(s) > 0 else "#D3D3D3"
        G.add_node(s_id, label=label, color=color, title=f"Abstract State with {len(s)} predicates")

    # Add edges to Graph
    for pre_s, post_s, prob, a_name in transitions:
        pre_id = get_state_id(pre_s)
        post_id = get_state_id(post_s)
        
        if verbosity == 0:
            edge_label = a_name
        else:
            edge_label = f"{a_name}\n(p={prob:.2f})"
            
        color = "green" if prob >= 0.8 else ("orange" if prob >= 0.2 else "red")
        
        G.add_edge(pre_id, post_id, label=edge_label, color=color)
        
    # Set up Abstract Transition Matrix logic
    # Prepare states and transitions for the matrix
    atm_states = [get_state_id(s) for s in states_list]
    atm_transitions = [(get_state_id(pre), get_state_id(post), prob) for pre, post, prob, _ in transitions]
    
    atm = AbstractTransitionMatrix(atm_states, atm_transitions)
    print("\n--- Abstract Transition Matrix ---")
    print("States:", atm_states)
    print(atm.get_transition_matrix())
    print(f"Spectral Gap: {atm.get_spectral_gap():.4f}")
    print("----------------------------------\n")

    # Layout for better structure
    output_name = os.path.basename(domain_path).replace(".ppddl", f"_markov_chain_v{verbosity}")
    print(f"Exporting Abstract Markov Chain to '{output_name}.html'...")
    
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.save_graph(f"{output_name}.html")
    print(f"Graph generated at: {os.path.abspath(f'{output_name}.html')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Abstract Markov Chain Graph")
    parser.add_argument("domain_path", help="Path to PPDDL domain file")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level for visualization (0=basic, 1=detailed)")
    
    args = parser.parse_args()
    generate_markov_chain(args.domain_path, verbosity=args.verbosity)
