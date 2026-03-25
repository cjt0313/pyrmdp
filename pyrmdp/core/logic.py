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

@dataclass
class ActionDefinition:
    name: str
    preconditions: List[str]
    effects: List[Dict[str, Union[float, List[str]]]]  # List of {"prob": float, "add": List[str], "del": List[str]}
    
def _extract_blocks(text: str) -> List[str]:
    blocks = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '(':
            if depth == 0:
                start = i
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:i+1])
    return blocks

def parse_ppddl_actions(filepath: str) -> List[ActionDefinition]:
    with open(filepath, 'r') as f:
        content = f.read()

    actions = []
    # Find all action blocks by splitting on "(:action"
    parts = content.split("(:action")
    if len(parts) <= 1:
        return actions

    for idx, part in enumerate(parts):
        if idx == 0: continue
        
        # Find the end of this action block by balancing brackets
        depth = 1 # We just stripped "(:action", so depth is 1
        end_idx = -1
        for i, char in enumerate(part):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            continue
            
        action_body = part[0:end_idx]
        
        # Name is the first word
        name_match = re.match(r'\s*([\w\-]+)', action_body)
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Preconditions
        preconditions = []
        pre_match = re.search(r':precondition\s+(.*?)(?=:effect|:parameters|$)', action_body, re.DOTALL | re.IGNORECASE)
        if pre_match:
            pre_text = pre_match.group(1)
            for p in re.findall(r'\(\s*([\w\-]+)', pre_text):
                pred = p.lower()
                if pred not in ['and', 'or', 'not', 'forall', 'exists']:
                    preconditions.append(pred)
        preconditions = list(set(preconditions)) # Unique
        
        # Effects
        effects = []
        eff_match = re.search(r':effect\s+(.*)', action_body, re.DOTALL | re.IGNORECASE)
        if eff_match:
            eff_text = eff_match.group(1).strip()
            
            is_prob = "probabilistic" in eff_text.lower()
            
            def parse_eff_block(block_text):
                add_eff = []
                del_eff = []
                
                # Match (not (predicate ...))
                for m in re.finditer(r'\(\s*not\s*\(\s*([\w\-]+)[^\)]*\)\s*\)', block_text, re.IGNORECASE):
                    del_eff.append(m.group(1).lower())
                
                # Remove not blocks
                stripped = re.sub(r'\(\s*not\s*\(\s*[\w\-]+[^\)]*\)\s*\)', '', block_text, flags=re.IGNORECASE)
                
                # Find all remaining predicates
                for p in re.findall(r'\(\s*([\w\-]+)', stripped):
                    pred = p.lower()
                    if pred not in ['and', 'or', 'probabilistic', 'when', 'not']:
                        add_eff.append(pred)
                        
                return add_eff, del_eff
            
            if is_prob:
                # remove (probabilistic ...) wrapper if exists
                if eff_text.startswith('('):
                    eff_text = eff_text[1:-1]
                eff_text = re.sub(r'^\s*probabilistic\s+', '', eff_text, flags=re.IGNORECASE)
                
                # Split by probability numbers
                tokens = re.split(r'([0-9.]+)\s*', eff_text)
                current_prob = 1.0
                for token in tokens:
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        current_prob = float(token)
                    except ValueError:
                        # it's a block
                        add_eff, del_eff = parse_eff_block(token)
                        effects.append({"prob": current_prob, "add": list(set(add_eff)), "del": list(set(del_eff))})
            else:
                add_eff, del_eff = parse_eff_block(eff_text)
                effects.append({"prob": 1.0, "add": list(set(add_eff)), "del": list(set(del_eff))})
                
        actions.append(ActionDefinition(name=name, preconditions=preconditions, effects=effects))
        
    return actions

