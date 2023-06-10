import networkx as nx
import numpy as np
import torch

from utils import *
from alignment.final import FINAL
from greedy_match import greedy_match
from hyperdimensional_similarity import hyperdimensional_similarity

if __name__ == "__main__":
    biogrid_path = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.220.tab3.txt"

    # Build the PPI graph.
    # G1 = build_network_from_biogrid_ppi(biogrid_path, 
    #                                     id_a=9606, 
    #                                     id_b=9606, 
    #                                     only_physical=True)

    G1 = build_network_from_edge_list("data/PPI.txt")

    # Get the adjacency matrix
    A1 = torch.from_numpy(nx.to_numpy_array(G1))

    # Build the partial synthetic network from G1
    P = generate_permutation_matrix(G1.number_of_nodes())    # permutation matrix

    # Compute new adjacency matrix
    A2 = P @ A1 @ P.T

    # Build the partial synthetic network
    G2 = build_network_from_adjacency_matrix(A2,
                                             node_names=None,
                                             remove_self_loops=True)
    
    # Get the groundtruth alignments
    P_non_zeros = list(torch.nonzero(P))
    groundtruth = [(i.item(), j.item()) for i, j in P_non_zeros]

    # Compute the  degree similarity network
    H = hyperdimensional_similarity(G1, G2, vector_size=1000)
    print(H)
    exit(0)
    
    # Run FINAL
    alpha = 0.4
    maxiter = 30
    tol = 1e-4
    S = FINAL(A1, A2, H, alpha, maxiter, tol)

    M = greedy_match(S)

    indices = np.flatnonzero(M)

    print(indices)
    
