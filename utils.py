import csv
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch


def set_reproducibility():
    """
    Set seeds and options for reproducibility.
    """
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def normalized_degree_similarity_network(n1, n2):
    """
    Given two networks compute the degree similarity network H.
    The network is normalized such that sum(sum(H)) = 1.
    """

    # Calculate degree sequences of n1 and n2
    n1_degrees = np.array([n1.degree[node] for node in n1.nodes()])
    n2_degrees = np.array([n2.degree[node] for node in n2.nodes()])

    # Compute the normalized degree network
    normalized_degree_network = np.zeros((len(n2.nodes()), len(n1.nodes())))

    for i, node_n2 in enumerate(n2.nodes()):
        for j, node_n1 in enumerate(n1.nodes()):
            neighbors_n1 = set(n1.neighbors(node_n1))
            neighbors_n2 = set(n2.neighbors(node_n2))
            common_neighbors = len(neighbors_n1.intersection(neighbors_n2))
            normalized_degree_network[i, j] = common_neighbors / (n1_degrees[j] + n2_degrees[i])

    # Normalize the network
    normalized_degree_network /= np.sum(normalized_degree_network)

    return normalized_degree_network


def generate_permutation_matrix(n):
    """
    Generate a random permutation matrix
    of size n x n.
    """
    perm = np.random.permutation(n)
    perm_matrix = np.eye(n)[perm]
    return torch.from_numpy(perm_matrix)


def build_network_from_edge_list(edgelist):
    """
    Reads the edge list file and buil the network.
    
    The edgelist must be provided as a tab-separated table. The
    first two columns of the table will be interpreted as an
    interaction gene1 <==> gene2.
    
    Lines that start with '#' will be ignored.
    """

    sniffer = csv.Sniffer()
    line_delimiter = None
    for line in open(edgelist, 'r'):
        if line[0] == '#':
            continue
        else:
            dialect = sniffer.sniff(line)
            line_delimiter = dialect.delimiter
            break
    if line_delimiter == None:
        print
        'edgelist format not correct'
        exit(0)

    # Read the network:
    G = nx.Graph()
    for line in open(edgelist, 'r'):
        # Lines starting with '#' will be ignored.
        if line[0] == '#':
            continue
        # The first two columns in the line will be interpreted as an
        # interaction gene1 <=> gene2
        # line_data   = line.strip().split('\t')
        line_data = line.strip().split(line_delimiter)
        node1 = line_data[0]
        node2 = line_data[1]
        G.add_edge(node1, node2)

    return G


def build_network_from_biogrid_ppi(data_path, id_a=None, id_b=None, only_physical=True, get_lcc=True):
    """
    Given the path for a biogrid database, build the corresponding PPI.
    If id_a and id_b are given, select only the interaction between the organism with ID A
    ant the organism with ID B.
    If only_physical is True, select only the physical interactions. 
    If get_lcc is True, use only the largest connected component.

    Return: the protein-protein interaction graph G.
    """

    # Read the database and build the DataFrame.
    df = pd.read_csv(data_path, sep="\t", header=0, low_memory=False)

    # Select according ID A and B.
    if id_a is not None and id_b is None:
        df = df.loc[
            (df["Organism ID Interactor A"] == id_a)]
    
    elif id_a is None and id_b is not None:
        df = df.loc[
            (df["Organism ID Interactor B"] == id_b)]
        
    elif id_a is not None and id_b is not None:
        df = df.loc[
            (df["Organism ID Interactor A"] == id_a) &
            (df["Organism ID Interactor B"] == id_b)]
    else:
        pass


    # Select only physical interactions if requested.
    if only_physical:
        df = df.loc[df["Experimental System Type"] == "physical"]
    
    # Build the graph.
    G = nx.from_pandas_edgelist(df,
        source = "Official Symbol Interactor A", 
        target = "Official Symbol Interactor B", 
        create_using=nx.Graph())  #nx.Graph doesn't allow duplicated edges

    # Remove self loops.
    self_loop_edges = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loop_edges)

    # Take only the LCC.
    if get_lcc:
        conn_comp = list(nx.connected_components(G))
        LCC = max(conn_comp, key=len)
        
        G = G.subgraph(LCC)
    
    return G


def build_network_from_adjacency_matrix(adj_matrix, node_names:dict=None, remove_self_loops=True):
    """
    Given an adjacency matrix build the currespondent graph.

    If node_names is not None, then use this dictionary to 
    assign the name of the nodes.
    """

    # Create an empty graph.
    G = nx.Graph()
    
    # Add weighted edges to the graph.
    if node_names:  # assign node names
        for i, node_i in node_names.items():
            for j, node_j in node_names.items():
                weight = adj_matrix[i, j]
                if weight != 0:
                    G.add_edge(node_i, node_j, weight=weight)
    else:
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                weight = adj_matrix[i, j]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)
    
    # Remove self loops.
    if remove_self_loops:
        self_loop_edges = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loop_edges)
    
    return G