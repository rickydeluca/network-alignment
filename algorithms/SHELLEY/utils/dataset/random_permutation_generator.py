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


def generate_random_permutation(
    graph, num_copies: int, p_remove: float = 0.0, p_add: float = 0.0
):
    """
    Generates `num_copies` copies of a graph with random permutations of nodes, edges removed, and new edges added.

    Args:
        graph (torch_geometric.data.Data): The original PyTorch Geometric graph.
        num_copies (int): The number of random permutations to generate.
        p_remove (float): Probability of removing an edge.
        p_add (float): Probability of adding a new edge.

    Returns:
        tuple: A tuple containing:
            - list: A list of PyTorch Geometric graphs representing the m random permutations.
            - list: A list of ground truth matrices corresponding to the mappings between nodes in the original and permuted graphs.
    """
    graphs = []
    groundtruth_matrices = []

    for _ in range(num_copies):
        permuted_nodes = np.random.permutation(graph.num_nodes)
        permuted_graph = graph.clone()
        permuted_graph.x[:, 0] = torch.tensor(permuted_nodes, dtype=torch.float)

        edge_index = to_networkx(permuted_graph).edges()
        edges_to_remove = [edge for edge in edge_index if np.random.rand() < p_remove]
        edges_to_add = [
            (i, j)
            for i in range(graph.num_nodes)
            for j in range(i + 1, graph.num_nodes)
            if np.random.rand() < p_add
        ]

        permuted_graph.edge_index = (
            torch.tensor(
                [edge for edge in edge_index if edge not in edges_to_remove]
                + edges_to_add,
                dtype=torch.long,
            ).t().contiguous()
        )

        graphs.append(permuted_graph)

        # Create ground truth matrix
        groundtruth_matrix = torch.zeros(
            (graph.num_nodes, graph.num_nodes), dtype=torch.long
        )
        for i, node in enumerate(permuted_nodes):
            groundtruth_matrix[i, node] = 1
        groundtruth_matrices.append(groundtruth_matrix)

    return graphs, groundtruth_matrices
