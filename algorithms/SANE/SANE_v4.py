"""
Learn embeddings and cross combine all in once.
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
from algorithms.SANE.sane_utils import (PairData, Data, get_embedding_model,
                                        get_mapping_model, networkx_to_pyg,
                                        read_network_dataset)
from utils.graph_utils import load_gt


class SANE(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        # Reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        super(SANE, self).__init__(source_dataset, target_dataset)

        self.seed = args.seed
        self.S = None

        # Unsupervised parameters
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
        self.hidden_channels = args.hidden_channels
        self.embedder = None
        self.mapper = None
        self.optimizer = None
        self.criterion = None

        # Suepervised parameters
        self.train_dict = args.train_dict
        self.groundtruth = args.groundtruth

        # Device
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
        
        self.preprocess_data()
        self.init_models()

        self.optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) +
            list(self.crosser.parameters()) + 
            list(self.mapper.parameters()),
            lr=self.lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.learn_embeddings()
        self.test_embeddings()

        self.predict_alignments()

        return self.S

    
    def supervised_learning(self):
        self.preprocess_supervised()
        self.init_mapping_model()
        self.mapping_optimizer = torch.optim.Adam(self.crosser.parameters(), lr = self.lr)
        self.mapping_criterion = torch.nn.BCEWithLogitsLoss()
        self.learn_mapping()
    

    def init_models(self):
        """
        Initialize the models for embedding and mapping.
        """

        self.embedder = get_embedding_model(
            self.embedding_model,
            in_channels=self.train_data.num_features,
            hidden_channels=self.hidden_channels,
            out_channels=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.embedding_dropout
        ).to(self.device)

        self.crosser = get_mapping_model(
            'cross_attention',
            k=self.embedding_dim,
            heads=2
        ).to(self.device)

        self.mapper = get_mapping_model(
            self.mapping_model,
            input_dim=self.embedding_dim,
            dropout=self.mapping_dropout
        ).to(self.device)



    def preprocess_data(self):
        # Read network datasets in a format suitable for pytorch geometric Data object
        data_s = read_network_dataset(self.source_dataset, pos_info=True).to(self.device)
        data_t = read_network_dataset(self.target_dataset, pos_info=True).to(self.device)

        print(f'data_s: {data_s}')
        print(f'data_t: {data_t}')

        # UNSUPERVISED LEARNING
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

        # - Group source and target networks to perform sampling
        train_data = Batch.from_data_list([train_data_s, train_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])
        val_data = Batch.from_data_list([val_data_s, val_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])
        test_data = Batch.from_data_list([test_data_s, test_data_t], follow_batch=['x', 'edge_weight', 'edge_attr', 'edge_label', 'edge_labels_index'])

        # Define LinkNeighbor dataloaders
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

        # Save everything globally
        self.data_s = data_s
        self.data_t = data_t
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_ln_loader = train_ln_loader
        self.val_ln_loader = val_ln_loader
        self.test_ln_loader = test_ln_loader
        
        # SUPERVISED LEARNING
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

            # Trasform to `edge_index` of shape (2, N) and use a subset of
            # training set as validation
            train_and_val_index = torch.tensor([list(train_dict.keys()), list(train_dict.values())]).to(self.device)
            test_index = torch.tensor([list(test_dict.keys()), list(test_dict.values())]).to(self.device)
            full_index = torch.tensor([list(full_dict.keys()), list(full_dict.values())]).to(self.device)

            split_size = int(0.8 * train_and_val_index.size(1))
            indices = torch.randperm(train_and_val_index.size(1))   # split randomly
            train_index = train_and_val_index[:, indices[:split_size]]
            val_index = train_and_val_index[:, indices[split_size:]]


            # Save globaly
            self.train_gt = train_gt
            self.test_gt = test_gt
            self.full_gt = full_gt
            self.train_dict = train_dict
            self.test_dict = test_dict
            self.full_dict = full_dict
            self.train_index = train_index
            self.val_index = val_index
            self.test_index = test_index
            self.full_index = full_index

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
            self.crosser.train()
            self.mapper.train()

            # UNSUPERVISED
            unsup_train_loss = 0.0
            unsup_train_examples = 0

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
                link_pred = self.mapper.pred(h_src, h_dst)

                # Backward step
                loss = self.criterion(link_pred, batch.edge_label)
                loss.backward()
                self.optimizer.step()

                unsup_train_loss += float(loss) * link_pred.numel()
                unsup_train_examples += link_pred.numel()

            unsup_train_loss /= unsup_train_examples

            # SUPERVISED
            sup_train_loss = 0.0
            sup_train_examples = 0
            self.optimizer.zero_grad()

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

            # Generate embeddings
            source_embeddings = self.embedder(self.data_s.x, self.data_s.edge_index).unsqueeze(0)
            target_embeddings = self.embedder(self.data_t.x, self.data_t.edge_index).unsqueeze(0)
            
            # Apply transformer with cross attention module
            y_source2target, y_target2source = self.crosser(source_embeddings, target_embeddings)

            # Get the crossed embeddings corresponding to the nodes in `train_aligns`
            y_source2target = y_source2target[:, train_aligns[0]].squeeze(0)
            y_target2source = y_target2source[:, train_aligns[1]].squeeze(0)

            # Compute cosine similarity
            pred_aligns = self.mapper.pred(y_source2target, y_target2source)

            # Backward step
            loss = self.criterion(pred_aligns, train_labels)
            loss.backward()
            self.optimizer.step()

            sup_train_loss += float(loss) * pred_aligns.numel()
            sup_train_examples += pred_aligns.numel()
            sup_train_loss /= sup_train_examples

            print(f"Unsupervised train loss: {unsup_train_loss:.4f}")
            print(f"Supervised train loss: {sup_train_loss:.4f}")

            # ==============
            #   VALIDATION
            # ==============

            # Setup validation
            self.embedder.eval()
            self.crosser.eval()
            self.mapper.eval()

            # UNSUPERVISED
            with torch.no_grad():
                unsup_val_loss = 0.0
                unsup_val_examples = 0

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
                    link_pred = self.mapper.pred(h_src, h_dst)

                    # Compute validation loss
                    loss = self.criterion(link_pred, batch.edge_label)
                    unsup_val_loss += float(loss) * link_pred.numel()
                    unsup_val_examples += link_pred.numel()

                unsup_val_loss /= unsup_val_examples

                # SUPERVISED
                sup_val_loss = 0.0
                sup_val_examples = 0

                # Sample negative alignments
                pos_val_aligns = self.val_index
                num_pos_val_aligns = pos_val_aligns.shape[1]
  
                neg_val_aligns = negative_sampling(
                    pos_val_aligns,
                    num_nodes=(
                        torch.unique(self.val_index[0]).shape[0],
                        torch.unique(self.val_index[1]).shape[0]),
                    num_neg_samples=num_pos_val_aligns * 2,   # Neg ratio of 2
                    force_undirected=True
                )      
                num_neg_val_aligns = neg_val_aligns.shape[1]

                # Generate labels (1: positive align, 0: negative align)
                val_aligns = torch.cat((pos_val_aligns, neg_val_aligns), dim=1)
                val_labels = torch.cat((torch.ones(num_pos_val_aligns),
                                        torch.zeros(num_neg_val_aligns)),
                                        dim=0)
                num_val_aligns = val_aligns.shape[1]

                # Shuffle alignment indices and labels
                permutation = torch.randperm(num_val_aligns)
                val_aligns = val_aligns[:, permutation].to(self.device)
                val_labels = val_labels[permutation].to(self.device)

                # Generate embeddings
                source_embeddings = self.embedder(self.data_s.x, self.data_s.edge_index).unsqueeze(0)
                target_embeddings = self.embedder(self.data_t.x, self.data_t.edge_index).unsqueeze(0)
            
                # Apply transformer with cross attention module
                y_source2target, y_target2source = self.crosser(source_embeddings, target_embeddings)

                # Get the crossed embeddings corresponding to the nodes in `train_aligns`
                y_source2target = y_source2target[:, val_aligns[0]].squeeze(0)
                y_target2source = y_target2source[:, val_aligns[1]].squeeze(0)

                # Compute cosine similarity
                pred_aligns = self.mapper.pred(y_source2target, y_target2source)

                # Compute loss step
                loss = self.criterion(pred_aligns, val_labels)
                sup_val_loss += float(loss) * pred_aligns.numel()
                sup_val_examples += pred_aligns.numel()
                sup_val_loss /= sup_val_examples

            print(f"Unsupervised val loss: {unsup_val_loss:.4f}")
            print(f"Supervised val loss: {sup_val_loss:.4f}")

            # Early Stopping
            if self.early_stop:
                if sup_val_loss < best_val_loss:
                    best_val_loss = sup_val_loss
                    best_model_state_dict = {   # Best model found till now.
                        'embedder': self.embedder.state_dict(),
                        'crosser': self.crosser.state_dict(),
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
            self.crosser.load_state_dict(best_model_state_dict['crosser'])
            self.mapper.load_state_dict(best_model_state_dict['mapper'])
            print(f"Best val loss: {best_val_loss:.4f}", end="\n\n")
    
    
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


    def predict_alignments(self):
        self.embedder.eval()
        self.mapper.eval()
        self.crosser.eval()
        
        with torch.no_grad():
            h_src = self.embedder(self.data_s.x, self.data_s.edge_index).unsqueeze(0)
            h_tgt = self.embedder(self.data_t.x, self.data_t.edge_index).unsqueeze(0)
            y_src, y_tgt = self.crosser(h_src, h_tgt)


        y_src = F.normalize(y_src.squeeze(0), p=2, dim=-1)
        y_tgt = F.normalize(y_tgt.squeeze(0), p=2, dim=-1)
        self.S = torch.mm(y_src, y_tgt.t()).detach().cpu().numpy()