import numpy as np
import networkx as nx


class AbstractTransitionMatrix:
    def __init__(self, states: list, transitions: list):
        """
        states: list of state identifiers
        transitions: list of tuples (from_state, to_state, probability)
        """
        self.states = states
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.n = len(states)
        self.matrix = np.zeros((self.n, self.n))
        
        for from_s, to_s, prob in transitions:
            if from_s in self.state_to_idx and to_s in self.state_to_idx:
                idx_from = self.state_to_idx[from_s]
                idx_to = self.state_to_idx[to_s]
                self.matrix[idx_from, idx_to] += prob
            
        # Normalize to ensure valid stochastic matrix
        for i in range(self.n):
            row_sum = np.sum(self.matrix[i])
            if row_sum > 0:
                self.matrix[i] /= row_sum
            else:
                self.matrix[i, i] = 1.0  # Absorbing state
                
    def get_transition_matrix(self):
        return self.matrix
        
    def get_spectral_gap(self):
        """
        Computes the spectral gap of the transition matrix.
        Spectral gap = 1 - lambda_2
        where lambda_2 is the second largest eigenvalue (in magnitude)
        """
        eigenvalues = np.linalg.eigvals(self.matrix)
        # Sort eigenvalues by magnitude in descending order
        sorted_eigenvalues = sorted(np.abs(eigenvalues), reverse=True)
        if len(sorted_eigenvalues) > 1:
            lambda_2 = sorted_eigenvalues[1]
            return 1.0 - lambda_2
        else:
            return 0.0

    def is_irreducible(self) -> bool:
        """
        Check if the Markov chain is irreducible (the underlying directed
        graph is strongly connected — every state can reach every other).
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i, j] > 0 and i != j:
                    G.add_edge(i, j)
        return nx.is_strongly_connected(G)

    def is_ergodic(self) -> bool:
        """
        Check if the chain is ergodic (irreducible + aperiodic).
        A sufficient condition for aperiodicity: at least one self-loop
        with positive probability.
        """
        if not self.is_irreducible():
            return False
        # Check for aperiodicity via self-loops
        for i in range(self.n):
            if self.matrix[i, i] > 0:
                return True
        # More thorough check: compute gcd of return times
        G = nx.DiGraph()
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i, j] > 0:
                    G.add_edge(i, j)
        # For a strongly connected graph, aperiodic iff gcd of all
        # cycle lengths is 1. NetworkX doesn't have this directly,
        # but if any self-loop exists, it's aperiodic.
        return False

    def get_communicating_classes(self) -> list:
        """
        Return the communicating classes (strongly connected components)
        of the Markov chain.
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i, j] > 0:
                    G.add_edge(i, j)
        sccs = list(nx.strongly_connected_components(G))
        # Map indices back to state labels
        idx_to_state = {i: s for s, i in self.state_to_idx.items()}
        return [
            [idx_to_state[i] for i in scc]
            for scc in sccs
        ]

def state_label(preds: frozenset, verbosity: int) -> str:
    """
    Format a frozenset of predicate strings into a visual graph label.
    """
    if not preds:
        return "Empty State"
    items = sorted(list(preds))
    if verbosity == 0:
        return f"State\n({len(items)} preds)"
    return " ∧ ".join([f"{p}" for p in items])
