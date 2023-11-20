import copy
import numpy as np
import networkx as nx


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


def generate_hypervector_basis(n_vectors: int, vector_size: int) -> np.ndarray:
    """
    Generate a basis of hypervectors.
    
    This function generates a matrix, where each row represents a hypervector.
    The elements of the hypervectors are randomly chosen from {-1, 1}.
    
    Args:
        n_vectors (int):    The number of hypervectors (rows) to generate.

        vector_size (int):  The size (dimensionality) of each hypervector
                            (columns).

    Returns:
        np.ndarray:         A 2D NumPy array of shape (n_vectors, vector_size) 
                            containing the generated hypervectors.
                            Each element is randomly chosen from {-1, 1}.
    """

    return np.random.choice([-1, 1], size=(n_vectors, vector_size))


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


def encode_network_hypervectors(G, basis, verbose=False):
    
    # Compute node features:
    # - Page Rank
    page_rank = convert_dict_negatives_to_zero(nx.pagerank(G, alpha=0.85))
    if verbose:
        print("============")
        print("  PageRank  ")
        print("============")
        print("Node\tPageRank")
        for node, rank in page_rank.items():
            print(f"{node}\t{rank:.4f}")
        print()

    # - Node Degrees
    node_degree = normalize_dict({node: degree for node, degree in G.degree()})
    if verbose:
        print("================")
        print("  Node Degrees  ")
        print("================")
        print("Node\tDegree")
        for node, degree in node_degree.items():
            print(f"{node}\t{degree:.4f}")
        print()

    # - Closeness Centralities
    closeness_centrality = normalize_dict(nx.closeness_centrality(G))
    if verbose:
        print("========================")
        print("  Closeness Centrality  ")
        print("========================")
        print("Node\Closeness")
        for node, value in closeness_centrality.items():
            print(f"{node}\t{value:.4f}")
        print()

    # - Betweenness Centrality
    betweenness_centrality = normalize_dict(nx.betweenness_centrality(G))
    if verbose:
        print("========================")
        print("  Betweennes Centrality ")
        print("========================")
        print("Node\Closeness")
        for node, value in betweenness_centrality.items():
            print(f"{node}\t{value:.4f}")
        print()

    # - Eigenvector Centrality 
    eigenvector_centrality = normalize_dict(nx.eigenvector_centrality(G))
    if verbose:
        print("========================")
        print("  Eigenvector Centrality ")
        print("========================")
        print("Node\Closeness")
        for node, value in eigenvector_centrality.items():
            print(f"{node}\t{value:.4f}")
        print()

    # - Clustering Coefficient
    clustering_coefficient = normalize_dict(nx.clustering(G))
    if verbose:
        print("========================")
        print("  Clusteing Coefficient ")
        print("========================")
        print("Node\Closeness")
        for node, value in clustering_coefficient.items():
            print(f"{node}\t{value:.4f}")
        print()

    # Compute the node hypervectors using basis hypervectors and node features
    _node_hypervectors = {node: flip_components(basis[0], rank) for node, rank in page_rank.items()}
    for node, value in node_degree.items():
        _node_hypervectors[node] += flip_components(basis[1], value)
    for node, value in closeness_centrality.items():
        _node_hypervectors[node] += flip_components(basis[2], value)
    for node, value in betweenness_centrality.items():
        _node_hypervectors[node] += flip_components(basis[3], value)
    for node, value in eigenvector_centrality.items():
        _node_hypervectors[node] += flip_components(basis[4], value)
    for node, value in clustering_coefficient.items():
        _node_hypervectors[node] += flip_components(basis[5], value)
    
    # Re-binarize the hypervectors
    node_hypervectors = {}
    for node, hypervector in _node_hypervectors.items():
        node_hypervectors[node] = binarize_hypervector(hypervector)

    # Output hypervectors
    if verbose:
        print("=====================")
        print("  Node Hypervectors  ")
        print("=====================")
        print("Node\tHypervectors")
        for node, hypervector in node_hypervectors.items():
            print(f"{node}\t{hypervector}")
        print()

    return node_hypervectors


if __name__ == "__main__":

    # Create a basis hypervector for each node feature
    node_features = ["page_rank", "node_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality", "clustering_coefficient"]
    basis = generate_hypervector_basis(len(node_features), 10000)

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

    node_hypervectors = encode_network_hypervectors(G, basis)

    # Check similarities
    print("Hypervector similarities:")
    print(f"A - A:\t{hypervector_similarity(node_hypervectors['A'], node_hypervectors['A'])}")
    print(f"A - B:\t{hypervector_similarity(node_hypervectors['A'], node_hypervectors['B'])}")
    print(f"B - A:\t{hypervector_similarity(node_hypervectors['B'], node_hypervectors['A'])}")
    print(f"A - F:\t{hypervector_similarity(node_hypervectors['A'], node_hypervectors['F'])}")
    print()