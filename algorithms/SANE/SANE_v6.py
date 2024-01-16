"""
Ricominciamo per l'ultima volta... di nuovo.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             roc_curve)
from torch_geometric.data import Batch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SANE.sane_utils import (get_embedding_model, get_mapping_model,
                                        get_prediction_model,
                                        read_network_dataset)
from utils.graph_utils import load_gt


class SANE(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        # Reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        super(SANE, self).__init__(source_dataset, target_dataset)

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_dict     = args.train_dict
        self.groundtruth    = args.groundtruth
        self.seed           = args.seed
        self.device         = torch.device(args.device)
        self.S              = None

        # Unsupervised parameters
        self.embedding_model    = args.embedding_model
        self.embedding_dim      = args.embedding_dim
        self.num_layers         = args.num_layers
        self.hidden_channels    = args.hidden_channels
        self.dropout_emb        = args.dropout_emb
        self.pos_info           = args.pos_info

        self.prediction_model   = args.prediction_model
        self.dropout_pred       = args.dropout_pred
        
        self.batch_size_emb   = args.batch_size_emb
        self.epochs_emb       = args.epochs_emb
        self.lr_emb           = args.lr_emb
        self.early_stop_emb   = args.early_stop_emb
        self.patience_emb     = args.patience_emb

        # Suepervised parameters
        self.mapping_model  = args.mapping_model
        self.heads          = args.heads

        self.batch_size_map = args.batch_size_map
        self.epochs_map     = args.epochs_map
        self.lr_map         = args.lr_map
        self.early_stop_map = args.early_stop_map
        self.patience_map   = args.patience_map


    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.S


    def align(self):
        """Performs alignment from source to target dataset."""

        self.learn_embeddings()
        self.learn_alignments()
        self.predict_alignments()

        return self.S
    

    def learn_embeddings(self):
        """
        Learn the node embeddings for both networks by training 
        in an unsupervised way for link predictions.
        """
        self.preprocess_embedding()
        self.init_embedding_models()
        self.optimizer_emb = torch.optim.Adam(list(self.embedder.parameters()) + list(self.predictor.parameters()), lr=self.lr_emb)
        self.criterion_emb = torch.nn.BCEWithLogitsLoss()
        self.train_embedding()
        self.test_embedding()

    
    def learn_alignments(self):
        """
        Given the learned embeddings for both networks,
        learn to alignate them using supervised learning and cross attention.
        """
        self.preprocess_mapping()
        self.init_mapping_model()
        self.optimizer_map = torch.optim.Adam(self.mapper.parameters(), lr = self.lr_map)
        self.criterion_map = torch.nn.BCEWithLogitsLoss()
        self.learn_mapping()
    

    def init_embedding_models(self):
        """
        Initialize the models for embedding and mapping.
        """

        self.embedder = get_embedding_model(
            self.embedding_model,
            in_channels=self.train_data.num_features,
            hidden_channels=self.hidden_channels,
            out_channels=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_emb
        ).to(self.device)

        self.predictor = get_prediction_model(
            self.prediction_model,
            input_dim=self.embedding_dim,
            dropout=self.dropout_pred
        ).to(self.device)


    def preprocess_embedding(self):
        # Read network datasets in a format suitable for pytorch geometric Data object
        data_s = read_network_dataset(self.source_dataset, pos_info=self.pos_info).to(self.device)
        data_t = read_network_dataset(self.target_dataset, pos_info=self.pos_info).to(self.device)

        print(f'data_s: {data_s}')
        print(f'data_t: {data_t}')

        # Split both source and target in train, test and val
        splitter = RandomLinkSplit(
            num_val=0.1,
            num_test=0.2,
            disjoint_train_ratio=0.0,
            add_negative_train_samples=False,    # it will be performed later
            is_undirected=True
        )
        
        train_data_s, val_data_s, test_data_s = splitter(data_s)
        train_data_t, val_data_t, test_data_t = splitter(data_t)

        # Group source and target networks to perform sampling
        train_data = Batch.from_data_list([train_data_s, train_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])
        val_data = Batch.from_data_list([val_data_s, val_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])
        test_data = Batch.from_data_list([test_data_s, test_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])

        # Define LinkNeighbor dataloaders
        train_loader_emb = LinkNeighborLoader(
            data=train_data,
            subgraph_type='bidirectional',
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=self.batch_size_emb,
            shuffle=True,
            num_workers=6,
            persistent_workers=True
        )

        val_loader_emb = LinkNeighborLoader(
            data=val_data,
            subgraph_type="bidirectional",
            num_neighbors=[20, 10],
            edge_label_index=val_data.edge_label_index,
            edge_label=val_data.edge_label,
            batch_size=self.batch_size_emb,
            shuffle=False
        )

        test_loader_emb = LinkNeighborLoader(
            data=test_data,
            subgraph_type="bidirectional",
            num_neighbors=[20, 10],
            edge_label_index=test_data.edge_label_index,
            edge_label=test_data.edge_label,
            batch_size=self.batch_size_emb,
            shuffle=False
        )

        # Save everything globally
        self.data_s             = data_s
        self.data_t             = data_t
        self.train_data         = train_data
        self.val_data           = val_data
        self.test_data          = test_data
        self.train_loader_emb   = train_loader_emb
        self.val_loader_emb     = val_loader_emb
        self.test_loader_emb    = test_loader_emb
        

    def train_embedding(self):
        """
        Train the models to generate the node embeddings
        """

        def _train_one_step():
            # Setup training    
            self.embedder.train()
            self.predictor.train()
            train_loss = 0.0
            train_examples = 0

            # Mini-batching
            for batch in tqdm(self.train_loader_emb, desc=f"Epoch {epoch} training:"):
                
                batch = batch.to(self.device)
                self.optimizer_emb.zero_grad()

                # Generate embeddings
                h = self.embedder(batch.x, batch.edge_index)
                
                # Predict links
                h_src = h[batch.edge_label_index[0]]
                h_dst = h[batch.edge_label_index[1]]
                link_pred = self.predictor.pred(h_src, h_dst)

                # Backward step
                loss = self.criterion_emb(link_pred, batch.edge_label)
                loss.backward()
                self.optimizer_emb.step()

                train_loss += float(loss) * link_pred.numel()
                train_examples += link_pred.numel()
            
            train_loss /= train_examples
            return train_loss
        

        def _val_one_step():
            # Setup validation
            self.embedder.eval()
            self.predictor.eval()
            val_loss = 0.0
            val_examples = 0

            with torch.no_grad():
                for batch in tqdm(self.val_loader_emb, desc=f"Epoch {epoch} validation:"):
                    batch = batch.to(self.device)

                    # Generate embeddings
                    h = self.embedder(batch.x, batch.edge_index)
                    
                    # Predict links
                    h_src = h[batch.edge_label_index[0]]
                    h_dst = h[batch.edge_label_index[1]]
                    link_pred = self.predictor.pred(h_src, h_dst)

                    # Compute validation loss
                    loss = self.criterion_emb(link_pred, batch.edge_label)
                    val_loss += float(loss) * link_pred.numel()
                    val_examples += link_pred.numel()

            val_loss /= val_examples
            return val_loss

        if self.early_stop_emb:
            patience = self.patience_emb
            best_val_loss = float('inf')
            patience_counter = 0
            best_emb_models = None

        # Training and validation loop
        for epoch in range(self.epochs_emb):
            print(f"Epoch {epoch+1}")
            
            train_loss = _train_one_step()
            val_loss = _val_one_step()

            print(f"\tTrain loss: {train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}", end="\n\n")

            # Early Stopping
            if self.early_stop_emb:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_emb_models = {   # Best model found till now
                        'embedder': self.embedder.state_dict(),
                        'predictor': self.predictor.state_dict()
                    }
                    patience_counter = 0
                    print("New best model found!")
                else:
                    patience_counter += 1
                    print(f"*** Patience counter: {patience_counter} ***")

                if patience_counter >= patience:
                    print("*** Early stopping triggered! ***", end="\n")
                    break
        
        # If trained with early stopping, load best models.
        if self.early_stop_emb:
            self.embedder.load_state_dict(best_emb_models['embedder'])
            self.predictor.load_state_dict(best_emb_models['predictor'])
            print(f"Best val loss: {best_val_loss:.4f}", end="\n\n")
    
    
    def test_embedding(self):

        # Setup test
        self.embedder.eval()
        self.predictor.eval()

        true_labels = []
        predicted_scores = []
        tp, fp, tn, fn = 0, 0, 0, 0
        test_loss = 0.0
        test_examples = 0

        # Test
        with torch.no_grad():
            for batch in tqdm(self.test_loader_emb, desc=f"Test:"):
                batch = batch.to(self.device)

                # Generate embeddings
                h = self.embedder(batch.x, batch.edge_index)
                
                # Predict links
                h_src = h[batch.edge_label_index[0]]
                h_dst = h[batch.edge_label_index[1]]
                link_pred = self.predictor.pred(h_src, h_dst)

                # Compute validation loss
                loss = self.criterion_emb(link_pred, batch.edge_label)
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

        print('\n\n')
        print("UNSUPERVISED LEARNING REPORT:")
        print(f"Test Loss:\t\t{test_loss:.4f}")
        print("------------------------------")
        print(f"Precision:\t\t{precision:.4f}")
        print(f"Recall:\t\t\t{recall:.4f}")
        print(f"F1 Score:\t\t{f1:.4f}")
        print(f"Accuracy:\t\t{accuracy:.4f}")
        print("------------------------------")
        print(f"Precision-Recall AUC:\t{pr_auc:.4f}")
        print(f"ROC-AUC:\t\t{roc_auc:.4f}")
        print("\n")


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


    def preprocess_mapping(self):
        """
        Prepare the dataset with the alignments to use for training and testing.
        """

        if self.train_dict is not None:
            # Load groundtruth alignmnets
            train_gt = load_gt(self.train_dict, self.source_dataset.id2idx, self.target_dataset.id2idx, 'dict')
            test_gt = load_gt(self.groundtruth, self.source_dataset.id2idx, self.target_dataset.id2idx, 'dict')
            full_gt = {}
            full_gt.update(train_gt)
            full_gt.update(test_gt)

            # Convert names to indices
            train_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in train_gt.items()}
            test_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in test_gt.items()}
            full_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in full_gt.items()}

            # Trasform to `edge_index` of shape (2, N)
            train_index = torch.tensor([list(train_dict.keys()), list(train_dict.values())]).to(self.device)
            test_index = torch.tensor([list(test_dict.keys()), list(test_dict.values())]).to(self.device)
            full_index = torch.tensor([list(full_dict.keys()), list(full_dict.values())]).to(self.device)

            self.train_gt = train_gt
            self.test_gt = test_gt
            self.full_gt = full_gt
            self.train_dict = train_dict
            self.test_dict = test_dict
            self.full_dict = full_dict
            self.train_index = train_index
            self.test_index = test_index
            self.full_index = full_index

    
    def init_mapping_model(self):
        self.embedder.eval()
        with torch.no_grad():
            self.source_embeddings = self.embedder(self.data_s.x, self.data_s.edge_index)
            self.target_embeddings = self.embedder(self.data_t.x, self.data_t.edge_index)

        self.mapper = get_mapping_model(
            self.mapping_model,
            embedding_dim=self.embedding_dim,
            source_embedding=self.source_embeddings,
            target_embedding=self.target_embeddings,
            heads=self.heads
        ).to(self.device)


    def learn_mapping(self):
        """
        Learn the mapping between source and target embeddings.
        """

        def _train_one_step():
            self.mapper.train()
            train_loss = 0.0
            self.optimizer_map.zero_grad()

            # Sample negative alignments
            pos_train_aligns = self.train_index
            num_pos_train_aligns = pos_train_aligns.shape[1]
  
            neg_train_aligns = negative_sampling(
                pos_train_aligns,
                num_nodes=(
                    torch.unique(self.train_index[0]).shape[0],
                    torch.unique(self.train_index[1]).shape[0]),
                num_neg_samples=num_pos_train_aligns * 2,   # Neg ratio of 2
                force_undirected=True
            )      
            num_neg_train_aligns = neg_train_aligns.shape[1]

            # Generate labels (1: positive align, 0: negative align)
            train_aligns = torch.cat((pos_train_aligns, neg_train_aligns), dim=1)
            train_labels = torch.cat((torch.ones(num_pos_train_aligns),
                                      torch.zeros(num_neg_train_aligns)),
                                      dim=0)
            num_train_aligns = train_aligns.shape[1]

            # Shuffle alignment indices and labels
            permutation = torch.randperm(num_train_aligns)
            train_aligns = train_aligns[:, permutation].to(self.device)
            train_labels = train_labels[permutation].to(self.device)
            
            # Compute mapping loss
            loss = self.mapper.loss(train_aligns[0], train_aligns[1])
            loss.backward()
            self.optimizer_map.step()

            return train_loss
        
        # Train loop
        for epoch in tqdm(range(self.epochs_map), desc="Mapping training:"):
            train_loss = _train_one_step()
        
        print(f"Mapping train loss: {train_loss:.4f}")


    def predict_alignments(self):
        self.embedder.eval()
        self.predictor.eval()
        self.mapper.eval()
        
        with torch.no_grad():
            if self.mapping_model == 'cross_transformer':
                self.source_after_mapping, self.target_after_mapping = self.mapper(self.source_embeddings, self.target_embeddings)
                S = torch.matmul(self.source_after_mapping, self.target_after_mapping.t())
            else:
                self.source_after_mapping = self.mapper(self.source_embeddings)
                S = torch.matmul(self.source_after_mapping, self.target_embeddings.t())

        self.S = S.detach().cpu().numpy()
