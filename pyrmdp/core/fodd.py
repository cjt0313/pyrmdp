from dataclasses import dataclass
from typing import Optional, Dict, List, Union
from .logic import Atom, Variable, Constant

@dataclass
class FODDNode:
    id: int
    query: Optional[Atom]
    high: Optional[int]
    low: Optional[int]
    is_leaf: bool
    value: Optional[float]

class FODDManager:
    def __init__(self, global_order: List[str]):
        """
        Global Order specifies the ranking of predicates for sorting (R4).
        E.g., ["clear", "on", "on-table", "holding"]
        """
        self.nodes: Dict[int, FODDNode] = {}
        self.unique_table: Dict[tuple, int] = {}
        self.next_id = 0
        self.global_order = global_order

    def _get_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def get_leaf(self, value: float) -> int:
        key = ("LEAF", value)
        if key in self.unique_table:
            return self.unique_table[key]

        node_id = self._get_id()
        node = FODDNode(node_id, None, None, None, True, value)
        self.nodes[node_id] = node
        self.unique_table[key] = node_id
        return node_id

    def compare_atoms(self, atom1: Atom, atom2: Atom) -> int:
        """
        Returns -1 if atom1 < atom2, 1 if atom1 > atom2, 0 if equal.
        """
        try:
            rank1 = self.global_order.index(atom1.predicate)
        except ValueError:
            rank1 = len(self.global_order)
            
        try:
            rank2 = self.global_order.index(atom2.predicate)
        except ValueError:
            rank2 = len(self.global_order)

        if rank1 != rank2:
            return -1 if rank1 < rank2 else 1

        # If predicates are the same, compare terms (variables)
        t1_str = str(atom1.terms)
        t2_str = str(atom2.terms)
        if t1_str != t2_str:
            return -1 if t1_str < t2_str else 1
            
        return 0

    def get_node(self, query: Atom, high: int, low: int) -> int:
        """
        Returns a node ID for a given query, high branch, and low branch.
        Implements R2 (Unique Table) to maintain canonical forms.
        Also implicitly helps with R4 if used recursively in reductions.
        """
        key = ("NODE", query, high, low)
        if key in self.unique_table:
            return self.unique_table[key]

        node_id = self._get_id()
        node = FODDNode(node_id, query, high, low, False, None)
        self.nodes[node_id] = node
        self.unique_table[key] = node_id
        return node_id
