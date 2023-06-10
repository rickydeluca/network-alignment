import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def similarity(v1, v2):
    """
    Given two D-hyperdimensional vectors v1 and v2,
    compute the similarity between them using the dot product.
    Then scale the result by D.

    Return: Similarity measure.
            Greater the value, greater the similarity.
    """
    assert v1.dim() == 1 and v2.dim() == 1
    assert v1.size() == v2.size()
    
    D = v1.size(0)

    return torch.dot(v1, v2) / D


def flip_dimensions(tensor, w):
    """
    Given a tensor of dimension D, select the components from (w * D) till D
    and flip them by multiplying by -1.
    
    Args:
        tensor: D-dimensional tensor.
        w:      float number in (0,1] representing the edge weight.
    
    Return:
        tensor with flipped dimensions.
    """
    # Get the shape of the input tensor
    D = tensor.shape[0]
    start_dim = abs(round(w * D))

    flipped_tensor = tensor.clone()
    flipped_tensor[start_dim:] *= -1

    return flipped_tensor


def softmax_normalization_dict(dictionary: dict):
    """
    Given a dictionary, apply the softmax normalization
    over its values.
    """
    # Extract the values from the dictionary
    values = np.array(list(dictionary.values()))

    # Apply softmax function
    softmax_values = np.exp(values) / np.sum(np.exp(values))

    # Create a new dictionary with the softmax-normalized values
    softmax_dict = {key: value for key, value in zip(dictionary.keys(), softmax_values)}

    return softmax_dict


def standard_normalization_dict(dictionary: dict, offset=1e-6):
    """
    Given a dictionary perform a standard min-max
    normalization over its values.
    """
    # Extract the values from the dictionary
    values = list(dictionary.values())

    # Find the minimum and maximum values
    min_val = min(values)
    max_val = max(values)

    # Calculate the range
    value_range = max_val - min_val

    # Perform the standard normalization
    normalized_dict = {}
    for key, value in dictionary.items():
        normalized_value = ((value - min_val) / value_range) * (1 - offset) + offset
        normalized_dict[key] = normalized_value

    return normalized_dict


def standard_normalization_tensor(matrix: torch.Tensor, offset=1e-6):
    """
    Given a torch tensor matrix, perform a standard min-max
    normalization over its values.
    """
    # Find the minimum and maximum values
    min_val = torch.min(matrix)
    max_val = torch.max(matrix)

    # Calculate the range
    value_range = max_val - min_val

    # Perform the standard normalization
    normalized_matrix = ((matrix - min_val) / value_range) * (1 - offset) + offset

    return normalized_matrix


def normalize_adjacency_matrix(adj_matrix: torch.Tensor):
    # Calculate the absolute values of the matrix
    abs_matrix = torch.abs(adj_matrix)

    # Normalize the matrix to the range [0, 1]
    normalized_matrix = abs_matrix / abs_matrix.max()

    return normalized_matrix
    

def similarity_threshold(degree, D, plot=False):
    """
    Given the degree of a node A and the hypervector dimensions D,
    compute the threshold value under which we can say that an other node 
    B is not connected to A.

    To do so, we generate two random gaussian distributions:
    the first relative to the case of B linked to A,
    and the other to the case B not linked to A.
    From these two distribution we create a theoretical ROC curve
    and from this curve we retrieve the threshold value.
    """
    # Generate random samples from two Gaussian distributions

    # Distribution 1 (No edge between A and B)
    mu1, sigma1 = 0, np.sqrt(degree / D)
    samples1 = np.random.normal(mu1, sigma1, 1000)

    # Distribution 2 (Edge between A and B)
    mu2, sigma2 = 1, np.sqrt((degree-1) / D)
    samples2 = np.random.normal(mu2, sigma2, 1000)

    # Combine the samples from both distributions
    all_samples = np.concatenate((samples1, samples2))
    labels = np.concatenate((np.zeros_like(samples1), np.ones_like(samples2)))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, all_samples)

    # Calculate AUC score
    auc = roc_auc_score(labels, all_samples)

    # Plot ROC curve
    if plot:
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()

    # Find the threshold value
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]

    return threshold


def node_hypervectors(G1, G2, vector_size=1000):
    # Generate the hypervectors for the ones of the attributes
    degree_1 = torch.randint(2, size=(vector_size,)) * 2 - 1
    closeness_centrality_1 = torch.randint(2, size=(vector_size,)) * 2 - 1
    betweenness_centrality_1 = torch.randint(2, size=(vector_size,)) * 2 - 1
    eigenvector_centrality_1 = torch.randint(2, size=(vector_size,)) * 2 - 1
    clustering_coefficient_1 = torch.randint(2, size=(vector_size,)) * 2 - 1

    # Compute the attributes for all nodes in both network and store them in a dictionary
    G1_degree = standard_normalization_dict({n: G1.degree(n, weight="weight") for n in G1.nodes()})
    G1_closeness_centrality = standard_normalization_dict(nx.closeness_centrality(G1, distance=None, wf_improved=True))
    G1_betweenness_centrality = standard_normalization_dict(nx.betweenness_centrality(G1, normalized=True, weight="weight", seed=42))
    G1_eigenvector_centrality = standard_normalization_dict(nx.eigenvector_centrality_numpy(G1, weight="weight", max_iter=50, tol=0))
    G1_clustering_coefficient = standard_normalization_dict(nx.clustering(G1, weight="weight"))

    G2_degree = standard_normalization_dict({n: G2.degree(n, weight="weight") for n in G2.nodes()})
    G2_closeness_centrality = standard_normalization_dict(nx.closeness_centrality(G2, distance=None, wf_improved=True) )
    G2_betweenness_centrality = standard_normalization_dict(nx.betweenness_centrality(G2, normalized=True, weight="weight", seed=42))
    G2_eigenvector_centrality = standard_normalization_dict(nx.eigenvector_centrality_numpy(G2, weight="weight", max_iter=50, tol=0))
    G2_clustering_coefficient = standard_normalization_dict(nx.clustering(G2, weight="weight"))

    # Compute the hypervector representation for each node attribute
    # and bind them to get the final node representation
    H1 = {}
    H2 = {}

    # Network 1
    for node in G1.nodes():
        # Get the normalized node attributes
        degree = float(G1_degree[node])
        closeness = float(G1_closeness_centrality[node])
        betweenness = float(G1_betweenness_centrality[node])
        eigenvector = float(G1_eigenvector_centrality[node])
        clustering = float(G1_clustering_coefficient[node])

        # Compute the currespondent hypervectors
        hyper_degree = flip_dimensions(degree_1, degree)
        hyper_closeness = flip_dimensions(closeness_centrality_1, closeness)
        hyper_betweenness = flip_dimensions(betweenness_centrality_1, betweenness)
        hyper_eigenvector = flip_dimensions(eigenvector_centrality_1, eigenvector)
        hyper_clustering = flip_dimensions(clustering_coefficient_1, clustering)

        # Compute final node representation
        H1[node] = hyper_degree * hyper_closeness * hyper_betweenness * hyper_eigenvector * hyper_clustering

    # Network 2
    for node in G2.nodes():
        # Get the normalized node attributes
        degree = float(G2_degree[node])
        closeness = float(G2_closeness_centrality[node])
        betweenness = float(G2_betweenness_centrality[node])
        eigenvector = float(G2_eigenvector_centrality[node])
        clustering = float(G2_clustering_coefficient[node])

        # Compute the currespondent hypervectors
        hyper_degree = flip_dimensions(degree_1, degree)
        hyper_closeness = flip_dimensions(closeness_centrality_1, closeness)
        hyper_betweenness = flip_dimensions(betweenness_centrality_1, betweenness)
        hyper_eigenvector = flip_dimensions(eigenvector_centrality_1, eigenvector)
        hyper_clustering = flip_dimensions(clustering_coefficient_1, clustering)

        # Compute final node representation
        H2[node] = hyper_degree * hyper_closeness * hyper_betweenness * hyper_eigenvector * hyper_clustering
    
    return H1, H2


def weight_hypervectors(G1, G2, vector_size=1000):
    # Generate the random hypervector for weight 1
    weight_1 = torch.randint(2, size=(vector_size,)) * 2 - 1

    # Generate the weight representations for both networks
    # and store them in a dictionary of the form: {edge: weight_hypervector}.
    W1 = {}
    W2 = {}

    for edge in G1.edges():
        # Get the edge weight
        source, dest = edge
        try:
            weight = float(G1[source][dest]["weight"])
        except:
            weight = 1

        # Get the currespondent hyperdimensional representation
        W1[edge] = flip_dimensions(weight_1, weight)

    for edge in G2.edges():
        # Get the edge weight
        source, dest = edge
        try:
            weight = float(G2[source][dest]["weight"])
        except:
            weight = 1

        # Get the currespondent hyperdimensional representation
        W2[edge] = flip_dimensions(weight_1, weight)
    
    return W1, W2


def memory_objects(G1=None, G2=None, H1=None, H2=None, W1=None, W2=None, vector_size=1000, refine_memory=True):
    # Create the memory objects for both networks
    M1 = {}
    M2 = {}

    for node in G1.nodes():
        M1[node] = torch.zeros((vector_size,), dtype=torch.long)
        for neighbor in G1.neighbors(node):
            # If don't find the edge (node, neighbor) try with (neighbor, node).
            # This is possible due undirectness of the graph.
            try:  
                M1[node] += H1[neighbor] * W1[node, neighbor]
            except:
                M1[node] += H1[neighbor] * W1[neighbor, node]

    for node in G2.nodes():
        M2[node] = torch.zeros((vector_size,), dtype=torch.long)
        for neighbor in G2.neighbors(node):
            try:
                M2[node] += H2[neighbor] * W2[node, neighbor]
            except:
                M2[node] += H2[neighbor] * W2[neighbor, node]

    # Refine memory
    if refine_memory:
        print("Refining memory network 1... ", end="")
        keep_refine = True

        while keep_refine == True:
            # At each iteration the algorithm try to stop the refinement setting the
            # flag to False, but if during the computation we finish in a
            # "refinement sceario", the flag will be setted again to True 
            # and we must do another loop.
            keep_refine = False

            for node_i in G1.nodes():
                # Compute the threshold
                degree_i = G1.degree[node_i]
                threshold = similarity_threshold(degree_i, vector_size)

                for node_j in G1.nodes():
                    # Compute the decision score
                    decision_score = similarity(M1[node_i], H1[node_j])

                    # Refine memory
                    if node_i != node_j:
                        if decision_score < threshold and ((node_i, node_j) in G1.edges()):
                            # print(f"Refining memory of node {node_i}...")
                            M1[node_i] += H1[node_j]
                            keep_refine = True
                        elif decision_score >= threshold and ((node_i, node_j) not in G1.edges()):
                            # print(f"Refining memory of node {node_i}...")
                            M1[node_i] -= H1[node_j]
                            keep_refine = True
                        else:
                            continue
        print("done!\n")

        print("Refining memory network 2... ", end="")
        keep_refine = True

        while keep_refine == True:

            keep_refine = False

            for node_i in G2.nodes():
                # Compute the threshold
                degree_i = G2.degree[node_i]
                threshold = similarity_threshold(degree_i, vector_size)

                for node_j in G2.nodes():
                    # Compute the decision score
                    decision_score = similarity(M2[node_i], H2[node_j])

                    # Refine memory
                    if node_i != node_j:
                        if decision_score < threshold and ((node_i, node_j) in G2.edges()):
                            # print(f"Refining memory of node {node_i}...")
                            M2[node_i] += H2[node_j]
                            keep_refine = True
                        elif decision_score >= threshold and ((node_i, node_j) not in G2.edges()):
                            # print(f"Refining memory of node {node_i}...")
                            M2[node_i] -= H2[node_j]
                            keep_refine = True
                        else:
                            continue
        print("done!")

    return M1, M2


def hyperdimensional_similarity(G1, G2, vector_size=1000):
    """
    Compute the hyperdimensional representation for the graphs G1 and G2
    and compute a similarity matrix between the hyperdimensional nodes
    of the two graphs.
    """

    H1, H2 = node_hypervectors(G1, G2, vector_size=vector_size)

    W1, W2 = weight_hypervectors(G1, G2, vector_size=vector_size)

    M1, M2 = memory_objects(G1=G1, G2=G2,
                            H1=H1, H2=H2,
                            W1=W1, W2=W2,
                            vector_size=vector_size,
                            refine_memory=True)

    # Init alignment matrix
    similarity_matrix = torch.zeros((G1.number_of_nodes(),
                                     G2.number_of_nodes()))

    # Fill the matrix with the similarity measures
    for i, n_i in enumerate(G1.nodes()):
        for j, n_j in enumerate(G2.nodes()):
            similarity_matrix[i,j] = similarity(M1[n_i], M2[n_j])

    # Apply softmax to normalize the similarity matrix
    normalized_matrix = torch.softmax(similarity_matrix.flatten(), dim=0)
    normalized_matrix = normalized_matrix.reshape(similarity_matrix.shape)


    return normalized_matrix