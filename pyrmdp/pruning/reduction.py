from typing import List, Dict, Tuple, Optional
from ..core.logic import Atom, Variable, Constant
from ..core.fodd import FODDManager, FODDNode

class SyntacticReducer:
    def __init__(self, manager: FODDManager):
        self.manager = manager
        
    def simplify(self, roots: List[int]) -> List[int]:
        """
        Applies R1 (Neglect) and R3 (Merge) to the given roots, using memoization.
        Path assignments track (atom, truth_value) pairs so R3 correctly follows
        the True or False branch depending on how the atom was reached.
        """
        memo = {}

        def _simplify(node_id: int, path_assignments: frozenset) -> int:
            cache_key = (node_id, path_assignments)
            if cache_key in memo:
                return memo[cache_key]

            node = self.manager.nodes[node_id]
            if node.is_leaf:
                return node_id

            # Apply R3: If query already tested on the path, follow the
            # branch consistent with the recorded truth assignment.
            for (atom, truth_val) in path_assignments:
                if atom == node.query:
                    if truth_val:
                        result = _simplify(node.high, path_assignments)
                    else:
                        result = _simplify(node.low, path_assignments)
                    memo[cache_key] = result
                    return result
                    
            high = _simplify(
                node.high,
                path_assignments | {(node.query, True)}
            )
            low = _simplify(
                node.low,
                path_assignments | {(node.query, False)}
            )

            # Apply R1: Neglect if high == low
            if high == low:
                memo[cache_key] = high
                return high

            new_node_id = self.manager.get_node(node.query, high, low)
            memo[cache_key] = new_node_id
            return new_node_id

        return [_simplify(r, frozenset()) for r in roots]

class StrongReducer:
    def __init__(self, manager: FODDManager, axioms: str):
        self.manager = manager
        import z3
        self.z3 = z3
        self.solver = z3.Solver()
        # parse custom axioms string into z3 constraints...
        
    def reduce_strong(self, root: int):
        pass # full implementation relies on Z3 + custom string parsing

def apply(op, manager: FODDManager, node1_id: int, node2_id: int) -> int:
    """
    Apply a binary operator (op) to two FODDs to create a new FODD.
    op: A function that takes two values and returns a value (e.g., lambda x, y: x + y).
    """
    memo = {}

    def _apply(n1_id, n2_id):
        if (n1_id, n2_id) in memo:
            return memo[(n1_id, n2_id)]

        n1 = manager.nodes[n1_id]
        n2 = manager.nodes[n2_id]

        # Case 1: Both are leaves
        if n1.is_leaf and n2.is_leaf:
            res_val = op(n1.value, n2.value)
            res_id = manager.get_leaf(res_val)
            memo[(n1_id, n2_id)] = res_id
            return res_id

        # Case 2: Determine root variable order (R4)
        if n1.is_leaf:
            # n2 is internal, so n2 is "higher" (processed first)
            val_branch = n1_id
            high_call = (val_branch, n2.high)
            low_call  = (val_branch, n2.low)
            query = n2.query
        elif n2.is_leaf:
            # n1 is internal, so n1 is higher
            val_branch = n2_id
            high_call = (n1.high, val_branch)
            low_call  = (n1.low, val_branch)
            query = n1.query
        else:
            # Both internal, compare queries
            cmp = manager.compare_atoms(n1.query, n2.query)
            if cmp == 0: # Same query
                high_call = (n1.high, n2.high)
                low_call  = (n1.low, n2.low)
                query = n1.query
            elif cmp < 0: # n1 < n2 (n1 is higher priority)
                high_call = (n1.high, n2_id)
                low_call  = (n1.low, n2_id)
                query = n1.query
            else: # n1 > n2 (n2 is higher priority)
                high_call = (n1_id, n2.high)
                low_call  = (n1_id, n2.low)
                query = n2.query

        # Recursive steps
        h_res = _apply(*high_call)
        l_res = _apply(*low_call)

        # Create/Find node (R2 is handled by manager)
        res_id = manager.get_node(query, h_res, l_res)
        
        # (Optional) Syntactic Simplification (R1) could be checked here
        # if h_res == l_res: return h_res 

        memo[(n1_id, n2_id)] = res_id
        return res_id

    return _apply(node1_id, node2_id)