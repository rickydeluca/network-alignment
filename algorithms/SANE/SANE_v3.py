"""
In this version of SANE the subgraphs relative to the training, validation
and test indices will not be generated. Instead, the computation will be 
performed using the whole adjacency matrices. However, the prediction and 
the negative sampling is still based on train, val and test indices.
"""

import numpy as np
import torch
from torch_geometric.utils import negative_sampling

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SANE.sane_utils import (get_embedding_model, get_mapping_model,
                                        networkx_to_pyg)
from utils.graph_utils import load_gt


class SANE(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        # Reproducibility.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        super(SANE, self).__init__(source_dataset, target_dataset)

        # Model parameters.
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.embedding_model = args.embedding_model
        self.mapping_model = args.mapping_model
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dim
        self.epochs = args.epochs
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.patience = args.patience
        self.embedding_dropout = args.embedding_dropout
        self.mapping_dropout = args.mapping_dropout
        self.num_layers = args.num_layers
        self.hidden_sizes = args.hidden_sizes
        self.seed = args.seed
        self.embedder = None
        self.mapper = None
        self.optimizer = None
        self.criterion = None


        # Alignment parameters.
        self.S = None
        self.train_dict = args.train_dict
        self.groundtruth = args.groundtruth

        # Device.
        if args.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            raise ValueError("Invalid device. Choose from: 'cpu' and 'cuda'.")


    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.S


    def align(self):
        """Performs alignment from source to target dataset."""

        # Init training models.
        self.init_models()
        self.optimizer = torch.optim.Adam(list(self.embedder.parameters()) + list(self.mapper.parameters()), lr=self.lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Preprocess data to create train, test and validation subnetworks.
        self.preprocess_data()

        # Train models.
        self.learn_embeddings()

        # Test learned embeddings.
        self.test_embeddings()

        # Populate alignment matrix.
        self.predict_alignments()

        return self.S
    

    def init_models(self):
        """
        Initialize the models for embedding and mapping.
        """

        # Get feature dimension.
        feature_dim1 = self.source_dataset.features.shape[1]
        feature_dim2 = self.target_dataset.features.shape[1]
        max_feature_dim = max(feature_dim1, feature_dim2)

        # Init models.
        self.embedder = get_embedding_model(
            self.embedding_model,
            in_channels=max_feature_dim,
            hidden_channels=self.hidden_sizes,
            out_channels=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.embedding_dropout
        ).to(self.device)

        self.mapper = get_mapping_model(
            self.mapping_model,
            input_dim=self.embedding_dim,
            dropout=self.mapping_dropout
        ).to(self.device)


    def preprocess_data(self):
        # Convert networks in a format suitable for PyG.
        self.source_data = networkx_to_pyg(self.source_dataset).to(self.device)
        self.target_data = networkx_to_pyg(self.target_dataset).to(self.device)

        # Split edges splitted in train, validation and test wrt the
        # given `train_dict`. If not given, split randomly.
        if self.train_dict is not None:
            train_and_val_gt = load_gt(self.train_dict, self.source_dataset.id2idx, self.target_dataset.id2idx, 'dict')
            test_gt = load_gt(self.groundtruth, self.source_dataset.id2idx, self.target_dataset.id2idx, 'dict')
            full_gt = {}
            full_gt.update(train_and_val_gt)
            full_gt.update(test_gt)

            # Convert node names to indices.
            train_and_val_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in train_and_val_gt.items()}
            test_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in test_gt.items()}
            full_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in full_gt.items()}

            # Transform to pytorch geometric edgelist of shape (2, N).
            self.full_index = torch.tensor([list(full_dict.keys()), list(full_dict.values())]).to(self.device)
            train_and_val_index = torch.tensor([list(train_and_val_dict.keys()), list(train_and_val_dict.values())]).to(self.device)
            split_idx = int(train_and_val_index.shape[1] * 0.7) # 30% of training as validation.
            self.train_index = train_and_val_index[:, :split_idx]
            self.val_index = train_and_val_index[:, split_idx:]
            self.test_index = torch.tensor([list(test_dict.keys()), list(test_dict.values())]).to(self.device)


    def learn_embeddings(self):
        """
        Train the models to generate the node embeddings
        """

        if self.early_stop:
            patience = self.patience
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state_dict = None

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            
            # TRAINING

            # Setup training.
            self.embedder.train()
            self.mapper.train()
            train_loss = 0.0
            train_examples = 0
            self.optimizer.zero_grad()

            # Get positive and negative alignments.
            pos_train_index = self.train_index
            neg_train_index = negative_sampling(
                pos_train_index,
                num_nodes=(
                    torch.unique(self.train_index[0]).shape[0],
                    torch.unique(self.train_index[1]).shape[0]),
                force_undirected=True
            )
            all_train_index = torch.cat((pos_train_index, neg_train_index), dim=1)

            # Get positive and negative labels.
            # [1, 1, ..., 1, 0, 0, ..., 0]
            all_train_labels = torch.cat((
                torch.ones(pos_train_index.shape[1]),
                torch.zeros(neg_train_index.shape[1])),
                dim=0
            )

            # Shuffle index and labels.
            permutation = torch.randperm(all_train_index.shape[1])
            all_train_index = all_train_index[:, permutation].to(self.device)
            all_train_labels = all_train_labels[permutation].to(self.device)

            # Generate embeddings.
            source_embeddings, target_embeddings = self.embedder(self.source_data, self.target_data)
        
            # Get only the embeddings corresponding to train index and align them by columns.
            source_embeddings = source_embeddings[all_train_index[0]]
            target_embeddings = target_embeddings[all_train_index[1]]

            # Predict alignments.
            pred_links = self.mapper.pred(source_embeddings, target_embeddings)
            
            # Backward step.
            loss = self.criterion(pred_links, all_train_labels)
            loss.backward()
            self.optimizer.step()

            train_loss += float(loss) * pred_links.numel()
            train_examples += pred_links.numel()
            train_loss /= train_examples

            print(f"\ttrain loss: {train_loss}")

            # VALIDATION

            # Setup validation.
            self.embedder.eval()
            self.mapper.eval()
            val_loss = 0.0
            val_examples = 0

            with torch.no_grad():
                # Get validation indices and labels.
                pos_val_index = self.val_index
                neg_val_index = negative_sampling(
                    pos_val_index,
                    num_nodes=(
                        torch.unique(self.val_index[0]).shape[0],
                        torch.unique(self.val_index[1]).shape[0]),
                    force_undirected=True
                )
                all_val_index = torch.cat((pos_val_index, neg_val_index), dim=1)
                all_val_labels = torch.cat((
                    torch.ones(pos_val_index.shape[1]),
                    torch.zeros(neg_val_index.shape[1])),
                    dim=0
                )

                # Shuffle them.
                permutation = torch.randperm(all_val_index.shape[1])
                all_val_index = all_val_index[:, permutation].to(self.device)
                all_val_labels = all_val_labels[permutation].to(self.device)

                # Generate embeddings.
                source_embeddings, target_embeddings = self.embedder(self.source_data, self.target_data)
            
                source_embeddings = source_embeddings[all_val_index[0]]
                target_embeddings = target_embeddings[all_val_index[1]]
                
                # Predict alignments.
                pred_links = self.mapper.pred(source_embeddings, target_embeddings)

                # Compute loss.
                loss = self.criterion(pred_links, all_val_labels)
                val_loss += float(loss) * pred_links.numel()
                val_examples += pred_links.numel()
                val_loss /= val_examples
            
            print(f"\tval loss: {val_loss}")


            # EARLY STOPPING
            if self.early_stop:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state_dict = {   # Best model found till now.
                        'embedder': self.embedder.state_dict(),
                        'mapper': self.mapper.state_dict()
                    }
                    patience_counter = 0
                    print("New best model found!")
                else:
                    patience_counter += 1
                    # print(f"*** Patience counter: {patience_counter} ***")

                if patience_counter >= patience:
                    print("*** Early stopping triggered! ***", end="\n")
                    break
        
        # If trained with early stopping, load best models.
        if self.early_stop:
            self.embedder.load_state_dict(best_model_state_dict['embedder'])
            self.mapper.load_state_dict(best_model_state_dict['mapper'])


    def get_subgraph(self, data, node_index):
        """
        Generate the subgraph of the `data` object using the node indices
        in `node_index` subset.
        Returns the subgraph and a dictionary that maps the global node 
        indices to the new indices in the subgraph
        """
        node_subset = torch.unique(node_index, sorted=True)
        full2sub = {node.item(): idx for idx, node in enumerate(node_subset)}
        subgraph = data.subgraph(node_subset)

        return subgraph, full2sub
    
    
    def test_embeddings(self):

        # Setup test.
        self.embedder.eval()
        self.mapper.eval()
        test_loss = 0.0
        test_examples = 0

        # Test.
        with torch.no_grad():
            # Get test indices and labels.
            pos_test_index = self.test_index
            neg_test_index = negative_sampling(
                pos_test_index,
                num_nodes=(
                    torch.unique(self.test_index[0]).shape[0],
                    torch.unique(self.test_index[1]).shape[0]),
                force_undirected=True
            )
            all_test_index = torch.cat((pos_test_index, neg_test_index), dim=1)
            all_test_labels = torch.cat((
                torch.ones(pos_test_index.shape[1]),
                torch.zeros(neg_test_index.shape[1])),
                dim=0
            )

            # Shuffling.
            permutation = torch.randperm(all_test_index.shape[1])
            all_test_index = all_test_index[:, permutation].to(self.device)
            all_test_labels = all_test_labels[permutation].to(self.device)

            # Generate embeddings.
            source_embeddings, target_embeddings = self.embedder(self.source_data, self.target_data)
            source_embeddings = source_embeddings[all_test_index[0]]
            target_embeddings = target_embeddings[all_test_index[1]]

            # Predict alignments.
            pred_links = self.mapper.pred(source_embeddings, target_embeddings)

            # Compute loss.
            loss = self.criterion(pred_links, all_test_labels)
            test_loss += float(loss) * pred_links.numel()
            test_examples += pred_links.numel()
            test_loss /= test_examples

            print(f"\ntest loss: {test_loss}")


    def predict_alignments(self):
        self.embedder.eval()
        self.mapper.eval()
        
        with torch.no_grad():
            # Compute node embedding for source and target networks.
            source_embeddings, target_embeddings = self.embedder(self.source_data, self.target_data)
            print("source_embeddings", source_embeddings.shape)
            print("target_embeddings", target_embeddings.shape)

            # Use broadcasting to create extended embeddings where each node in
            # the source network is paired with each node in the target network.
            N = source_embeddings.shape[0]
            M = target_embeddings.shape[0]
            alignment_matrix = np.zeros((N, M))

            # Prediction only on test dataset
            for source_idx in self.test_index[0]:
                source_embeddings_extended = []
                target_embeddings_extended = []

                for target_idx in self.test_index[1]:
                    source_embeddings_extended.append(source_embeddings[source_idx])
                    target_embeddings_extended.append(target_embeddings[target_idx])

                source_embeddings_extended = torch.stack(source_embeddings_extended)
                target_embeddings_extended = torch.stack(target_embeddings_extended)
                
                pred_links = self.mapper.pred(source_embeddings_extended, target_embeddings_extended)
                pred_probs = torch.sigmoid(pred_links).detach().cpu().numpy()

                for j, target_idx in enumerate(self.test_index[1]):
                    alignment_matrix[source_idx, target_idx] = pred_probs[j]

            print("alignment_matrix: ", alignment_matrix)
            self.S = alignment_matrix

