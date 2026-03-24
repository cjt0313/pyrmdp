from dataclasses import dataclass
from typing import List, Union, Dict
import re

@dataclass
class Term:
    name: str

@dataclass
class Variable(Term):
    def __hash__(self):
        return hash(f"?{self.name}")
    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name
    def __str__(self):
        return f"?{self.name}"

@dataclass
class Constant(Term):
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return isinstance(other, Constant) and self.name == other.name
    def __str__(self):
        return self.name

@dataclass
class Atom:
    predicate: str
    terms: List[Union[Variable, Constant]]

    def substitute(self, mapping: dict):
        new_terms = []
        for t in self.terms:
            if isinstance(t, Variable) and t.name in mapping:
                new_terms.append(mapping[t.name])
            else:
                new_terms.append(t)
        return Atom(self.predicate, new_terms)

    def __hash__(self):
        return hash((self.predicate, tuple(self.terms)))
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.predicate == other.predicate and self.terms == other.terms

    def __str__(self):
        terms_str = ", ".join(str(t) for t in self.terms)
        return f"{self.predicate}({terms_str})"

def parse_ppddl_predicates(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the :predicates block
    match = re.search(r'\(:predicates\s*(.*?)\n\s*\)', content, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
        
    predicates_block = match.group(1)
    
    # Extract predicate names
    predicates = []
    for line in predicates_block.split('\n'):
        # match (predicate ?v1 ?v2)
        p_match = re.search(r'\(\s*([\w\-]+)', line)
        if p_match:
            predicates.append(p_match.group(1))
            
    return predicates
