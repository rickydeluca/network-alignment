import argparse
import numpy as np

from algorithms.HDA.hyperdimensional_encoding import *
from algorithms.network_alignment_model import NetworkAlignmentModel

from input.dataset import Dataset

class HDA(NetworkAlignmentModel):
    def __init__(self, source_dataset=None, target_dataset=None, node_features=None, vector_size=10000):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.node_features = node_features
        self.vector_size = vector_size

    def align(self):
        
        # Get networks
        G1 = self.source_dataset.G
        G2 = self.target_dataset.G

        # Define hypervector basis for node features
        # node_features = ["page_rank", "node_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality", "clustering_coefficient"]
        node_basis = generate_hypervector_basis(features=self.node_features, size=self.vector_size)

        # Get node hypervectors for both networks
        H1 = encode_node_hypervectors(
            G1,
            basis=node_basis,
            verbose=False
        )
        
        H2 = encode_node_hypervectors(
            G2,
            basis=node_basis,
            verbose=False
        )

        # # Get weight hypervectors
        # weight_basis = generate_hypervector_basis(features=["weight"], size=self.vector_size)
        # W1 = encode_weight_hypervectors(
        #     G1,
        #     basis=weight_basis
        # )

        # W2 = encode_weight_hypervectors(
        #     G2,
        #     basis=weight_basis
        # )

        # # Get memory hypervectors (combine the node hypervectors with the neighbour hypervectors)
        # M1 = encode_memory_hypervectors(G1, H1, W1)
        # M2 = encode_memory_hypervectors(G2, H2, W2)

        # Build the alignment matrix
        n1 = G1.nodes()         # Nodes first network
        n2 = G2.nodes()         # Nodes second network
        S = np.zeros((G1.number_of_nodes(), G2.number_of_nodes()))

        for i, node_i in enumerate(n1):
            for j, node_j in enumerate(n2):
                hypervector_i = H1[node_i]
                hypervector_j = H2[node_j]
                S[i, j] = hypervector_similarity(hypervector_i, hypervector_j)

        self.alignment_matrix = S
        return self.alignment_matrix


    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix



def parse_args():
    parser = argparse.ArgumentParser(description="HDA")
    parser.add_argument('--prefix1', default="dataspace/douban/online/graphsage")
    parser.add_argument('--prefix2', default="dataspace/douban/offline/graphsage")
    parser.add_argument('--node_features', nargs='+', default=["page_rank", "node_degree", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality", "clustering_coefficient"])
    parser.add_argument('--groundtruth', default=None)
    parser.add_argument('--vector_size', default=10000)

    return parser.parse_args()


def main(args):

    model = HDA(
        source_dataset=Dataset(args.prefix1),
        target_dataset=Dataset(args.prefix2),
        node_features=args.node_features,
        vector_size=args.vector_size
    )
    
    S = model.align()

    print(S)

if __name__ == "__main__":
    args = parse_args()
    main(args)






