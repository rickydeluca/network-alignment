import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected


def read_edge_list(file_path: str, weighted: bool=False):
    """
    Reads an edge list from a file.

    Args:
        file_path (str): The path to the edge list file.
        weighted (bool, optional): Whether the edge list contains weighted edges. Default is False.

    Returns:
        list: A list of edges, where each edge is represented as a tuple. If weighted is True, each tuple is of the form (node1, node2, weight).
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        if weighted:
            edges = [tuple(map(float, line.strip().split())) for line in lines]
        else:
            edges = [tuple(map(int, line.strip().split())) for line in lines]
    return edges


def compute_local_metric(graph: nx.Graph, metric_name: str):
    """
    Computes a local metric for each node in the graph.

    Args:
        graph (networkx.Graph): The input graph.
        metric_name (str): The name of the metric to compute.

    Returns:
        dict: A dictionary where keys are node indices and values are the computed metric values.
    """
    if metric_name == "degree":
        return dict(graph.degree())
    elif metric_name == "page_rank":
        return nx.pagerank(graph)
    elif metric_name == "assortativity":
        return nx.degree_assortativity_coefficient(graph)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")


def create_pyg_graph(edge_list, metric_name=None, undirected=True):
    """
    Creates a PyTorch Geometric graph from an edge list.

    Args:
        edge_list (list): A list of edges, where each edge is represented as a tuple of two integers.
        metric_name (str, optional): The name of the local metric to compute as node features. Default is None.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric graph.
    """
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    if undirected:
        # Ensure the graph is undirected
        edge_index = to_undirected(edge_index)

    num_nodes = int(edge_index.max()) + 1

    if metric_name:
        # Compute local metric
        nx_graph = to_networkx(Data(edge_index=edge_index), to_undirected=True)
        local_metric_values = np.array(
            list(compute_local_metric(nx_graph, metric_name).values())
        )

        # Normalize the metric values to be in the range [0, 1]
        local_metric_values = (local_metric_values - np.min(local_metric_values)) / (
            np.max(local_metric_values) - np.min(local_metric_values)
        )

        x = torch.tensor(local_metric_values, dtype=torch.float).view(-1, 1)
    else:
        x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index)


def add_and_remove_edges(G, p_new_connection, p_remove_connection):    
    """
    For each node, add a new connection to a random other node,
    with prob p_new_connection, remove a connection,
    with prob p_remove_connection.

    Returns the new graph (a copy) with added/removed edges and the ground truth matrix
    """               
    # Create a deep copy of the graph to avoid modifying the source graph
    new_G = nx.Graph(G)

    new_edges = []    
    rem_edges = []  

    for i, node in enumerate(new_G.nodes()):    
        # Find the other nodes this one is connected to    
        connected = [to for (fr, to) in new_G.edges(node)]    
        # And find the remainder of nodes, which are candidates for new edges   
        unconnected = [n for n in new_G.nodes() if n not in connected]    

        # Probabilistically add a random edge    
        if len(unconnected):  # Only try if a new edge is possible    
            if random.random() < p_new_connection:    
                new = random.choice(unconnected)    
                new_G.add_edge(node, new)    
                print("\tnew edge:\t {} -- {}".format(node, new))    
                new_edges.append((node, new))    
                # Book-keeping, in case both add and remove done in the same cycle  
                unconnected.remove(new)    
                connected.append(new)

        # Probabilistically remove a random edge    
        if len(connected):  # Only try if an edge exists to remove    
            if random.random() < p_remove_connection:    
                remove = random.choice(connected)    
                new_G.remove_edge(node, remove)    
                print("\tedge removed:\t {} -- {}".format(node, remove))    
                rem_edges.append((node, remove))    
                # Book-keeping, in case lists are important later    
                connected.remove(remove)    
                unconnected.append(remove)    

    return new_G, new_edges, rem_edges


# Test functions
if __name__ == '__main__':
    # Create an initial graph
    G_src = nx.erdos_renyi_graph(20, 0.15, seed=42)
    print("Source graph:")
    print(G_src)

    # Add/remove connections
    p_new_connection = 0.1
    p_remove_connection = 0.1

    G_tgt, new_edges, rem_edges = add_and_remove_edges(
        G_src, p_new_connection, p_remove_connection
    )

    print("\nAdded Edges:")
    print(new_edges)
    print("\nRemoved Edges:")
    print(rem_edges)
    print("New graph:")
    print(G_tgt)

    # Shuffle ID and index of nodes in target graph

