#!/usr/bin/env python3
import sys
import os
from pyrmdp.core.logic import Variable, Constant, Atom, parse_ppddl_predicates
from pyrmdp.core.fodd import FODDManager
from pyrmdp.vis.visualization import plot_fodd_structure

def generate_add_diagram(domain_path: str):
    """
    Parses a PPDDL domain file and generates a representative First-Order Decision Diagram (ADD)
    visualizing the logical structure of precondition checks (e.g., for an action reward).
    """
    print(f"Parsing domain file: {domain_path}")
    predicates = parse_ppddl_predicates(domain_path)
    
    if not predicates:
        print("Error: No predicates found in domain file.")
        return

    # Initialize FODD Manager with global order from predicates
    manager = FODDManager(global_order=predicates)
    
    # Generic variables for schema visualization
    loc = Variable("loc")
    obj = Variable("obj")
    
    # Values
    val_true = manager.get_leaf(1.0)
    val_false = manager.get_leaf(0.0)

    # Strategy: Build a naive ADD that includes ALL predicates in a linear chain.
    # This visualizes the FULL STATE SPACE decomposition order.
    # Root -> Pred1 -> Pred2 -> ... -> Leaves
    
    current_node = val_true
    
    # We iterate in reverse to build from bottom-up
    for pred_name in reversed(predicates):
        # Create a generic atom for this predicate
        # Heuristic: if pred is 'at', use [obj, loc]. Else use [obj]
        if pred_name in ["at", "on", "connected", "adjacent"]:
            terms = [obj, loc]
        elif pred_name in ["arm-at", "clear"]:
            terms = [loc]
        else:
            terms = [obj]
            
        atom = Atom(pred_name, terms)
        
        # Create a node: IF pred THEN continue_up ELSE 0
        # (This implies a conjunctive "AND" logic for all predicates)
        current_node = manager.get_node(atom, current_node, val_false)

    output_name = os.path.basename(domain_path).replace(".ppddl", "_add_diagram")
    print(f"Generated ADD Root ID: {current_node}")
    print(f"Exporting ADD diagram to '{output_name}.html'...")
    plot_fodd_structure(manager, current_node, output_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_add.py <path_to_domain.ppddl>")
        sys.exit(1)
    
    generate_add_diagram(sys.argv[1])
