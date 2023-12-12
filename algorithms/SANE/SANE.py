"""
Learn embeddings with unsupervised learning, 
then cross combine the network embeddings 
using a transformer block and use supervised learning on them.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             roc_curve)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SANE.sane_utils import (PairData, get_embedding_model,
                                        get_mapping_model, networkx_to_pyg,
                                        read_network_dataset)
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

        # Preprocess data for unsupervised learning
        self.preprocess_data()

        # Init unsupervised learning models
        self.init_models()
        self.optimizer = torch.optim.Adam(list(self.embedder.parameters()) + list(self.mapper.parameters()), lr=self.lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Train models and learn embeddings for both networks
        self.learn_embeddings()

        # Test learned embeddings
        self.test_embeddings()
        return
        # Populate alignment matrix.
        self.predict_alignments()

        return self.S
    

    def init_models(self):
        """
        Initialize the models for embedding and mapping.
        """

        # Get feature dimension.
        # feature_dim1 = self.source_dataset.features.shape[1]
        # feature_dim2 = self.target_dataset.features.shape[1]
        # max_feature_dim = max(feature_dim1, feature_dim2)

        # Init models.
        self.embedder = get_embedding_model(
            self.embedding_model,
            in_channels=self.train_data.num_features,
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
        # Read network datasets in a format suitable for pytorch geometric Data object
        data_s = read_network_dataset(self.source_dataset, pos_info=True)
        data_t = read_network_dataset(self.target_dataset, pos_info=True)

        print(f'data_s: {data_s}')
        print(f'data_t: {data_t}')

        # Prepare for unsupervised learning:
        # - Split both source and target in train, test and val
        splitter = RandomLinkSplit(
            num_val=0.1,
            num_test=0.2,
            disjoint_train_ratio=0.0,
            add_negative_train_samples=False,    # it will be performed later
            is_undirected=True
        )
        
        train_data_s, val_data_s, test_data_s = splitter(data_s)
        train_data_t, val_data_t, test_data_t = splitter(data_t)

        # print(f'train_data_s: {train_data_s}')
        # print(f'val_data_s: {val_data_s}')
        # print(f'test_data_s: {test_data_s}')

        # - Group source and target networks to perform sampling
        train_data = Batch.from_data_list([train_data_s, train_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])
        val_data = Batch.from_data_list([val_data_s, val_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])
        test_data = Batch.from_data_list([test_data_s, test_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])

        # - Define LinkNeighbor dataloaders
        train_ln_loader = LinkNeighborLoader(
            data=train_data,
            subgraph_type='bidirectional',
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            persistent_workers=True
        )

        val_ln_loader = LinkNeighborLoader(
            data=val_data,
            subgraph_type="bidirectional",
            num_neighbors=[20, 10],
            edge_label_index=val_data.edge_label_index,
            edge_label=val_data.edge_label,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_ln_loader = LinkNeighborLoader(
            data=test_data,
            subgraph_type="bidirectional",
            num_neighbors=[20, 10],
            edge_label_index=test_data.edge_label_index,
            edge_label=test_data.edge_label,
            batch_size=self.batch_size,
            shuffle=False
        )

        # - Save everything globally
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_ln_loader = train_ln_loader
        self.val_ln_loader = val_ln_loader
        self.test_ln_loader = test_ln_loader
        

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
            
            # ============
            #   TRAINING
            # ============
            # Setup training    
            self.embedder.train()
            self.mapper.train()
            train_loss = 0.0
            train_examples = 0

            # Mini-batching
            for batch in tqdm(self.train_ln_loader, desc=f"Epoch {epoch} training:"):
                
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Generate embeddings
                if self.embedding_model == 'sage':
                    h = self.embedder(batch.x, batch.edge_index)
                else:
                    raise("Only GraphSAGE was implemented within this SANE version!")
                
                # Predict links
                h_src = h[batch.edge_label_index[0]]
                h_dst = h[batch.edge_label_index[1]]
                # link_pred = (h_src * h_dst).sum(dim=-1) # Inner product
                link_pred = self.mapper.pred(h_src, h_dst)

                # Backward step
                loss = self.criterion(link_pred, batch.edge_label)
                loss.backward()
                self.optimizer.step()

                train_loss += float(loss) * link_pred.numel()
                train_examples += link_pred.numel()


            print(f"\tTrain loss: {train_loss/train_examples:.4f}")

            # ==============
            #   VALIDATION
            # ==============

            # Setup validation
            self.embedder.eval()
            self.mapper.eval()
            val_loss = 0.0
            val_examples = 0

            with torch.no_grad():
                for batch in tqdm(self.val_ln_loader, desc=f"Epoch {epoch} validation:"):
                    batch = batch.to(self.device)

                    # Generate embeddings
                    if self.embedding_model == 'sage':
                        h = self.embedder(batch.x, batch.edge_index)
                    else:
                        raise("Only GraphSAGE was implemented within this SANE version!")
                    
                    # Predict links
                    h_src = h[batch.edge_label_index[0]]
                    h_dst = h[batch.edge_label_index[1]]
                    # link_pred = (h_src * h_dst).sum(dim=-1)
                    link_pred = self.mapper.pred(h_src, h_dst)

                    # Compute validation loss
                    loss = self.criterion(link_pred, batch.edge_label)
                    val_loss += float(loss) * link_pred.numel()
                    val_examples += link_pred.numel()

            val_loss /= val_examples
            print(f"Validation loss: {val_loss:.4f}", end="\n\n")

            # Early Stopping
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
            print(f"Best val loss: {best_val_loss:.4f}", end="\n\n")


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

        # Setup test
        self.embedder.eval()
        self.mapper.eval()

        true_labels = []
        predicted_scores = []
        tp, fp, tn, fn = 0, 0, 0, 0
        test_loss = 0.0
        test_examples = 0

        # Test
        with torch.no_grad():
            for batch in tqdm(self.test_ln_loader, desc=f"Test:"):
                batch = batch.to(self.device)

                # Generate embeddings
                if self.embedding_model == 'sage':
                    h = self.embedder(batch.x, batch.edge_index)
                else:
                    raise("Only GraphSAGE was implemented within this SANE version!")
                
                # Predict links
                h_src = h[batch.edge_label_index[0]]
                h_dst = h[batch.edge_label_index[1]]
                # link_pred = (h_src * h_dst).sum(dim=-1)
                link_pred = self.mapper.pred(h_src, h_dst)

                # Compute validation loss
                loss = self.criterion(link_pred, batch.edge_label)
                test_loss += float(loss) * link_pred.numel()
                test_examples += link_pred.numel()

                # Convert logits to probabilities using sigmoid and store them
                predicted_probs = torch.sigmoid(link_pred).cpu().numpy()
                true_labels.extend(batch.edge_label.cpu().numpy())
                predicted_scores.extend(predicted_probs)

                # Class predictions using a threshold of 0.5
                preds_class = (predicted_probs > 0.5).astype(int)
                test_labels_class = batch.edge_label.cpu().numpy()

                tp += np.sum((preds_class == 1) & (test_labels_class == 1))
                fp += np.sum((preds_class == 1) & (test_labels_class == 0))
                tn += np.sum((preds_class == 0) & (test_labels_class == 0))
                fn += np.sum((preds_class == 0) & (test_labels_class == 1))
        
        # Compute metrics
        test_loss /= test_examples
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        # Compute Precision-Recall AUC
        precision_array, recall_array, _ = precision_recall_curve(true_labels, predicted_scores)
        pr_auc = auc(recall_array, precision_array)

        # Compute ROC curve and ROC-AUC
        fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
        roc_auc = roc_auc_score(true_labels, predicted_scores)

        print(f"Test Loss: {test_loss:.4f}")
        print("-----------------")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("----------------------------")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")


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

