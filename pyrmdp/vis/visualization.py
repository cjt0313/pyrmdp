import networkx as nx
from pyvis.network import Network
from ..core.fodd import FODDManager

def plot_fodd_structure(manager: FODDManager, root_id: int, title: str):
    """
    Export an interactive HTML graph of the FODD (R2 ensures canonical DAG).
    """
    G = nx.DiGraph()
    queue = [root_id]
    visited = set()

    while queue:
        curr_id = queue.pop(0)
        if curr_id in visited:
            continue
        visited.add(curr_id)

        node = manager.nodes[curr_id]
        label = f"{curr_id}"
        if node.is_leaf:
            label += f" ({node.value})"
            G.add_node(curr_id, label=label, shape="box")
        else:
            q_str = str(node.query)
            label += f"\n{q_str}"
            G.add_node(curr_id, label=label, shape="ellipse")

            if node.high is not None:
                G.add_edge(curr_id, node.high, label="True", color="green")
                queue.append(node.high)
            if node.low is not None:
                G.add_edge(curr_id, node.low, label="False", color="red")
                queue.append(node.low)

    net = Network(notebook=True)
    net.from_nx(G)
    net.show(f"{title}.html")
