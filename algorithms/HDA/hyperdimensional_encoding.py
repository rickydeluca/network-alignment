import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score


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



def normalize_vector(array):
    """
    Normalize an array of numbers to make 
    their sum equal to 1, similar to a probability distribution.

    Parameters:
        array (list or np.ndarray):     An array of numerical values.

    Returns:
        np.ndarray:                     An array of normalized values.
    """
    # Ensure the array is a NumPy array
    array = np.array(array)
    
    # Check for negative values
    if np.any(array < 0):
        raise ValueError("Array should not contain negative values.")
    
    # Calculate the sum of the array
    array_sum = np.sum(array)
    
    # Check for zero sum to avoid division by zero
    if array_sum == 0:
        raise ValueError("Sum of array must not be zero to avoid division by zero.")
    
    # Normalize the array by dividing each element by the sum
    normalized_array = array / array_sum
    
    return normalized_array


def normalize_dict(data_dict):
    """
    Normalize a dictionary of numbers to make  their sum equal to 1,
    similar to a probability distribution.

    Args:
        data_dict (dict):   A dictionary of numerical values.

    Returns:
        dict:               A dictionary of normalized values.
    """
    # Ensure the input is a dictionary
    if not isinstance(data_dict, dict):
        raise TypeError("Input should be a dictionary.")
    
    # Check for negative values
    if any(val < 0 for val in data_dict.values()):
        raise ValueError("Dictionary should not contain negative values.")
    
    # Calculate the sum of the values
    values_sum = sum(data_dict.values())
    
    # Check for zero sum to avoid division by zero
    if values_sum == 0:
        raise ValueError("Sum of values must not be zero to avoid division by zero.")
    
    # Normalize the values by dividing each value by the sum
    normalized_dict = {key: val / values_sum for key, val in data_dict.items()}
    
    return normalized_dict



def convert_dict_negatives_to_zero(input_dict):
    """
    Convert all negative values in a dictionary to zeros, returning a new dictionary.

    Args:
        input_dict (dict):  The input dictionary.

    Returns:
        dict:               A new dictionary with no negative values.
    """
    # Ensure the input is a dictionary
    if not isinstance(input_dict, dict):
        raise ValueError("Input must be a dictionary")
    
    # Create a copy of the input dictionary to avoid modifying the original
    output_dict = copy.deepcopy(input_dict)
    
    # Iterate through the dictionary and update negative values
    for key, value in output_dict.items():
        if isinstance(value, (int, float)) and value < 0:  # Check if the value is numeric and negative
            output_dict[key] = 0  # Update negative values to zero
        elif isinstance(value, dict):  # Check if the value is a nested dictionary
            output_dict[key] = convert_dict_negatives_to_zero(value)  # Recursively update nested dictionary
    
    return output_dict


def generate_hypervector_basis(features=None, size: int=None):
    """
    Generate a basis of hypervectors.
    
    This function generates a matrix, where each row represents a hypervector.
    The elements of the hypervectors are randomly chosen from {-1, 1}.
    
    Args:
        features:           The node features for which generate a basis.

        size (int):         The size of each basis.

    Returns:
        dict:               A dictionary in which for each basis is associated
                            its hyperdimensional basis.
    """

    basis = {}
    for feat in features:
        basis[feat] = np.random.choice([-1, 1], size=size)

    return basis


def flip_components(hypervector: np.ndarray, value: float) -> np.ndarray:
    """
    Flip the components of the D-dimensional `hypervector` from position
    (`value` x D) till position D.

    Args:
        hypervector (np.ndarray):   The input hypervector.

        values (float):             A real value between 0 and 1.
    
    Returns:
        np.ndarray:                 The hypervectors with fixed components.
    """

    # Ensure the input value is within the acceptable range
    if value < 0 or value > 1:
        raise ValueError("The input value must be between 0 and 1.")
    
    # Ensure the input hypervector is a NumPy array
    if not isinstance(hypervector, np.ndarray):
        raise TypeError("The input hypervector must be a NumPy array.")
    
    # Ensure the hypervector contains only -1 and 1 values
    if not np.all(np.isin(hypervector, [-1, 1])):
        raise ValueError("The hypervector must contain only -1 and 1 values.")
    
    # Calculate the position from which to flip the components
    D = len(hypervector)
    flip_start_pos = int(value * D)
    
    # Flip the components from flip_start_pos till position D
    flipped_hypervector = hypervector.copy()
    flipped_hypervector[flip_start_pos:] *= -1
    
    return flipped_hypervector


def hypervector_similarity(h1: np.ndarray, h2: np.ndarray):
    """
    Compute the similarity between two hypervectors.

    This function calculates the similarity between two hypervectors by computing
    the normalized dot product of the two. The normalization is done by dividing
    the dot product by the size of the hypervectors.

    Args:
        h1 (np.ndarray):    The first hypervector.
        h2 (np.ndarray):    The second hypervector.

    Returns:
        float:              The similarity between hypervector `h1` and `h2`.

    Raises:
        ValueError:         If the hypervectors do not have the same size.
    """
    # Ensure that both hypervectors have the same size
    if h1.shape != h2.shape:
        raise ValueError("The hypervectors do not have the same size.")
    
    # Get the size of the hypervectors
    size = h1.shape[0]

    return np.dot(h1, h2) / size


def binarize_hypervector(hypervector):
    """
    Transforms a hypervector with values -1, 0, and 1 to a hypervector with values -1 and 1.
    
    Args:
        hypervector (np.ndarray):   The input hypervector with values
                                    -1, 0, and 1.
    
    Returns:
        np.ndarray:                 The transformed hypervector with 
                                    values -1 and 1.
    """

    # Create a copy of the input hypervector to avoid modifying the original
    transformed_hypervector = np.copy(hypervector)

    # Keep only signs (-1, 0 and 1)
    transformed_hypervector = np.sign(transformed_hypervector)

    # Replace 0s with 1s
    transformed_hypervector[transformed_hypervector == 0] = 1
    
    return transformed_hypervector


def encode_node_hypervectors(G, basis=None, verbose=False):
    
    # Get feature names
    feature_names = basis.keys()

    # Compute features and store them in a dictionary
    feature_dict = {}

    # - Page Rank
    if 'page_rank' in feature_names:
        page_rank = convert_dict_negatives_to_zero(nx.pagerank(G, alpha=0.85))
        feature_dict['page_rank'] = page_rank

        if verbose:
            print("============")
            print("  PageRank  ")
            print("============")
            print("Node\tPageRank")
            for node, rank in page_rank.items():
                print(f"{node}\t{rank:.4f}")
            print()

    # - Node Degrees
    if 'node_degree' in feature_names:
        node_degree = normalize_dict({node: degree for node, degree in G.degree()})
        feature_dict['node_degree'] = node_degree

        if verbose:
            print("================")
            print("  Node Degrees  ")
            print("================")
            print("Node\tDegree")
            for node, degree in node_degree.items():
                print(f"{node}\t{degree:.4f}")
            print()

    # - Closeness Centralities
    if 'closeness_centrality' in feature_names:
        closeness_centrality = normalize_dict(nx.closeness_centrality(G))
        feature_dict['closeness_centrality'] = closeness_centrality

        if verbose:
            print("========================")
            print("  Closeness Centrality  ")
            print("========================")
            print("Node\Closeness")
            for node, value in closeness_centrality.items():
                print(f"{node}\t{value:.4f}")
            print()

    # - Betweenness Centrality
    if 'betweenness_centrality' in feature_names:
        betweenness_centrality = normalize_dict(nx.betweenness_centrality(G))
        feature_dict['betweenness_centrality'] = betweenness_centrality

        if verbose:
            print("========================")
            print("  Betweennes Centrality ")
            print("========================")
            print("Node\Closeness")
            for node, value in betweenness_centrality.items():
                print(f"{node}\t{value:.4f}")
            print()

    # - Eigenvector Centrality
    if 'eigenvector_centrality' in feature_names:
        eigenvector_centrality = normalize_dict(nx.eigenvector_centrality(G))
        feature_dict['eigenvector_centrality'] = eigenvector_centrality

        if verbose:
            print("========================")
            print("  Eigenvector Centrality ")
            print("========================")
            print("Node\Closeness")
            for node, value in eigenvector_centrality.items():
                print(f"{node}\t{value:.4f}")
            print()

    # - Clustering Coefficient
    if 'clustering_coefficient' in feature_names:
        clustering_coefficient = normalize_dict(nx.clustering(G))
        feature_dict['clustering_coefficient'] = clustering_coefficient

        if verbose:
            print("========================")
            print("  Clusteing Coefficient ")
            print("========================")
            print("Node\Closeness")
            for node, value in clustering_coefficient.items():
                print(f"{node}\t{value:.4f}")
            print()

    # Init the node hypervectors dictionary
    nodes = next(iter(feature_dict.values())).keys()
    node_hypervectors = {node: np.ones_like(next(iter(basis.values()))) for node in nodes}

    # Combine the feature hypervectors
    for feat in feature_names:
        for node, value in feature_dict[feat].items():
            node_hypervectors[node] *= flip_components(basis[feat], value)
    
    # Re-binarize the hypervectors
    bin_node_hypervectors = {}
    for node, hypervector in node_hypervectors.items():
        bin_node_hypervectors[node] = binarize_hypervector(hypervector)

    # Output hypervectors
    if verbose:
        print("=====================")
        print("  Node Hypervectors  ")
        print("=====================")
        print("Node\tHypervectors")
        for node, hypervector in bin_node_hypervectors.items():
            print(f"{node}\t{hypervector}")
        print()

    return bin_node_hypervectors


def encode_weight_hypervectors(G, basis):
    weight_hypervectors = {}

    for edge in G.edges():

        # Get the edge weight
        source, dest = edge

        # If is weighted add only the 'source, dest' edge
        if nx.is_weighted(G, edge=edge):
            weight = float(G[source][dest]["weight"])

            if nx.is_directed(G):
                weight_hypervectors[(source, dest)] = flip_components(basis["weight"], weight)
            else:
                weight_hypervectors[(source, dest)] = flip_components(basis["weight"], weight)
                weight_hypervectors[(dest, source)] = flip_components(basis["weight"], weight)

        # Otherwise, add alos 'dest, source'
        else: 
            if nx.is_directed(G):
                weight_hypervectors[(source, dest)] = basis["weight"]
            else:
                weight_hypervectors[(source, dest)] = basis["weight"]
                weight_hypervectors[(dest, source)] = basis["weight"]



    return weight_hypervectors


def encode_memory_hypervectors(G, node_hypervectors, weight_hypervectors, n_iters=100, refine=False):
    memory_hypervectors = {}

    # Generate memory hypervectors
    for node in G.nodes():
        memory_hypervectors[node] = node_hypervectors[node]

        for neighbor in G.neighbors(node):
            
            # If directed, sum only the incomning edge
            if nx.is_directed(G):   
                if (neighbor, node) in G.edges:
                    memory_hypervectors[node] += node_hypervectors[neighbor] * weight_hypervectors[neighbor, node]
                else:
                    # print(f"Not found edge ({neighbor}, {node})")
                    continue
            
            # Otherwise, sum all edges
            else:
                if (node, neighbor) in G.edges or (neighbor, node) in G.edges:
                    try:
                        memory_hypervectors[node] += node_hypervectors[neighbor] * weight_hypervectors[node, neighbor]
                    except:
                        memory_hypervectors[node] += node_hypervectors[neighbor] * weight_hypervectors[neighbor, node]
                else:
                    print(f"Not found edge ({node}, {neighbor})")
                    continue
                    
    # Refine memory
    vector_size = next(iter(memory_hypervectors.values())).shape[0]
    keep_refine = refine
    for _ in tqdm(range(n_iters), ascii=True, desc="Refining memory:"):
        
        # Check early stopping flag
        if keep_refine == False:
            break

        # At each iteration the algorithm try to stop the refinement setting 
        # the flag to False, but if during the computation we finish in a 
        # "refinement sceario", the flag will be setted again to True 
        # and we must perform another loop.
        keep_refine = False

        for node_i in G.nodes:
            # Compute the threshold
            degree_i = G.degree[node_i]
            threshold = similarity_threshold(degree_i, vector_size)

            for node_j in G.nodes:
                # Compute the decision score
                decision_score = hypervector_similarity(memory_hypervectors[node_i], node_hypervectors[node_j])

                # Refine memory
                if node_i != node_j:
                    if decision_score < threshold and ((node_i, node_j) in G.edges):
                        # print(f"Refining memory of node {node_i}...")
                        memory_hypervectors[node_i] += node_hypervectors[node_j]
                        keep_refine = True
                    elif decision_score >= threshold and ((node_i, node_j) not in G.edges):
                        # print(f"Refining memory of node {node_i}...")
                        memory_hypervectors[node_i] -= node_hypervectors[node_j]
                        keep_refine = True
                    else:
                        continue

    # Re-binarize memory hypervectors
    for node, hypervector in memory_hypervectors.items():
        memory_hypervectors[node] = binarize_hypervector(hypervector)

    return memory_hypervectors



if __name__ == "__main__":

    # Create a basis hypervector for each node feature and weight
    node_features = ["page_rank", "node_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality", "clustering_coefficient"]
    basis = generate_hypervector_basis(node_features, 10000)
    weight_basis = generate_hypervector_basis(["weight"], 10000)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'A')
    G.add_edge('D', 'B')
    G.add_edge('E', 'B')
    G.add_edge('E', 'D')
    G.add_edge('E', 'F')
    G.add_edge('F', 'C')
    G.add_edge('F', 'D')

    H = encode_node_hypervectors(G, basis, verbose=True)
    W = encode_weight_hypervectors(G, weight_basis)
    M = encode_memory_hypervectors(G, H, W)

    # Check similarities
    print("Hypervector similarities:")
    print(f"A - A:\t{hypervector_similarity(M['A'], M['A'])}")
    print(f"A - B:\t{hypervector_similarity(M['A'], M['B'])}")
    print(f"B - A:\t{hypervector_similarity(M['B'], M['A'])}")
    print(f"A - F:\t{hypervector_similarity(M['A'], M['F'])}")
    print()