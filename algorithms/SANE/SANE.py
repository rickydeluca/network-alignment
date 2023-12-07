import argparse

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SANE.sane_utils import (get_embedding_model, get_mapping_model,
                                        networkx_to_pyg)
from input.dataset import Dataset
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
        self.embedding_dropout = args.embedding_dropout
        self.mapping_dropout = args.mapping_dropout
        self.num_layers = args.num_layers
        self.hidden_sizes = args.hidden_sizes
        self.seed = args.seed

        # Alignment parameters.
        self.S = None
        self.embedder = None
        self.mapper = None
        self.optimizer = None
        self.criterion = None

        # Device.
        if args.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            raise ValueError("Invalid device. Choose from: 'cpu' and 'cuda'.")
    
        # Load train and test groundtruths.
        train_gt = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        test_gt = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

        # Build the full groundtruth and substitute the node 
        # names with the node indices.
        self.full_gt = {}
        self.full_gt.update(train_gt)
        self.full_gt.update(test_gt)
        self.full_gt = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in self.full_gt.items()}

        # Use a subset of the training dictionary to obtain a validation 
        # dictionary, finally convert train, val and test dictionaries
        # into edgelist tensors of shape (2, N), where N is the numeber of nodes.
        self.train_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in train_gt.items()}
        self.test_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in test_gt.items()}

        train_and_val_index = torch.tensor([list(self.train_dict.keys()), list(self.train_dict.values())]).to(self.device)
        split_idx = int(train_and_val_index.shape[1] * 0.8) # 20% of training as validation.
        self.train_index = train_and_val_index[:, :split_idx]
        self.val_index = train_and_val_index[:, split_idx:]
        self.test_index = torch.tensor([list(self.test_dict.keys()), list(self.test_dict.values())]).to(self.device)

        # [DEBUG]
        # print("train_index_shape: ", self.train_index.shape)
        # print("val_index_shape: ", self.val_index.shape)
        # print("test_index_shape: ", self.test_index.shape)


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
        self.learn_embeddings(patience=10)

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
        source_data = networkx_to_pyg(self.source_dataset).to(self.device)
        target_data = networkx_to_pyg(self.target_dataset).to(self.device)

        # Split in train, validation and test.
        source_data_train = source_data.subgraph(self.train_index[0])
        source_data_val = source_data.subgraph(self.val_index[0])
        source_data_test = source_data.subgraph(self.test_index[0])

        target_data_train = target_data.subgraph(self.train_index[1])
        target_data_val = target_data.subgraph(self.val_index[1])
        target_data_test = target_data.subgraph(self.test_index[1])

        self.source_data = source_data
        self.source_data_train = source_data_train
        self.source_data_val = source_data_val
        self.source_data_test = source_data_test

        self.target_data = target_data
        self.target_data_train = target_data_train
        self.target_data_val = target_data_val
        self.target_data_test = target_data_test

        # Create a mapping between global node indices and their position in the subgraphs
        # (i.e. the node `57` corresponds to the 42th node in the 'source train subgraph')
        self.global2train_source = {global_idx.item(): i for i, global_idx in enumerate(self.train_index[0])}
        self.global2val_source = {global_idx.item(): i for i, global_idx in enumerate(self.val_index[0])}
        self.global2test_source = {global_idx.item(): i for i, global_idx in enumerate(self.test_index[0])}
        
        self.global2train_target = {global_idx.item(): i for i, global_idx in enumerate(self.train_index[1])}
        self.global2val_target = {global_idx.item(): i for i, global_idx in enumerate(self.val_index[1])}
        self.global2test_starget = {global_idx.item(): i for i, global_idx in enumerate(self.test_index[1])}

        source_data.train_index = self.train_index[0]
        source_data.val_index = self.val_index[0]
        source_data.test_index = self.test_index[0]

        target_data.train_index = self.train_index[1]
        target_data.val_index = self.val_index[1]
        target_data.test_index = self.test_index[1]

        print("source_data:", source_data.train_index)
        print("target_data: ", target_data.train_index)
        return

        # [DEBUG]
        # print("Global to subgraph example: ", self.global2train_source)

    def learn_embeddings(self, patience=None):
        """
        Train the models to generate the node embeddings
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None

        for epoch in range(self.epochs):
            
            # TRAINING

            # Setup training.
            self.embedder.train()
            self.mapper.train()
            train_loss = 0.0
            train_examples = 0
            self.optimizer.zero_grad()

            # Sample negative alignments within each training iteration to have 2 classes to discriminate. 
            # Given the list of true cross network alingments we generate a list of the same lenght with false alignment.
            neg_train_index = negative_sampling(self.train_index, force_undirected=True)

            train_labels = torch.cat((torch.ones(self.train_index.shape[1]),     # [1, 1, ..., 1, 0, 0, ..., 0]
                                      torch.zeros(neg_train_index.shape[1])),
                                      dim=0).type(torch.LongTensor)
            
            all_train_index = torch.cat((self.train_index, neg_train_index),
                                        dim=1)
            
            # Shuffle index and labels.
            permutation = torch.randperm(all_train_index.shape[1])
            shuffled_all_train_index = all_train_index[:, permutation]
            shuffled_train_labels = train_labels[permutation]

            # [DEBUG]
            # print("permuataion: ", permutation)
            # print("shuffled_all_train_index shape: ", shuffled_all_train_index.shape)
            # print(f"shuffled_all_train_labels min: {torch.min(shuffled_all_train_index)}, max: {torch.max(shuffled_all_train_index)}")
            # print("shuffled_train_labels unique labels: ", torch.unique(shuffled_train_labels))
        
            # Generate embeddings.
            source_embeddings, target_embeddings = self.embedder(self.source_data_train, self.target_data_train)
        
            # Take only the embeddings corresponding to the nodes in `all_train_index` so that we can try to alignate
            # # them source to target and compare the prediction with the groundtruth labels.
            print("global2train_source:", self.global2train_source)
            print("Train source subgraph: ", self.source_data_train)
            print("Source train edge index", self.source_data_train.edge_index)
            source_local_train_index = torch.tensor([self.global2train_source[idx.item()] for idx in shuffled_all_train_index[0]])
            target_local_train_index = torch.tensor([self.global2train_target[idx.item()] for idx in shuffled_all_train_index[1]])
            source_embeddings = source_embeddings[source_local_train_index]
            target_embeddings = target_embeddings[target_local_train_index]

            print("source_local_train_index: ", source_local_train_index)
            print("target_loacl_train_index: ", target_local_train_index)
            return

            # Predict alignments.
            pred_links = self.mapper.pred(source_embeddings, target_embeddings)

            # Backward step.
            loss = self.criterion(pred_links, torch.ones(pred_links.shape[0], dtype=torch.long, device=self.device))
            loss.backward()
            self.optimizer.step()

            train_loss += float(loss) * pred_links.numel()
            train_examples += pred_links.numel()
            train_loss /= train_examples

            print(f"train loss: {train_loss}")

            # VALIDATION

            # Setup validation.
            self.embedder.eval()
            self.mapper.eval()
            val_loss = 0.0
            val_examples = 0

            with torch.no_grad():
                # Generate embeddings.
                source_embeddings, target_embeddings = self.embedder(self.source_data_val, self.target_data_val)

                # Predict alignments.
                pred_links = self.mapper.pred(source_embeddings, target_embeddings)

                # Compute loss.
                loss = self.criterion(pred_links, 0)
                val_loss += float(loss) * pred_links.numel()
                val_examples += pred_links.numel()
                val_loss /= val_examples


            # EARLY STOPPING

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = {   # Best model found till now.
                    'embedder': self.embedder.state_dict(),
                    'mapper': self.mapper.state_dict()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                # print(f"*** Patience counter: {patience_counter} ***")

            if patience_counter >= patience:
                print("*** Early stopping triggered! ***", end="\n")
                break
        
        # Load best parameters.
        self.embedder.load_state_dict(best_model_state_dict['embedder'])
        self.mapper.load_state_dict(best_model_state_dict['mapper'])


    """
    def train_unsup(self, network='source'):
        # Convert the NetworkX to PyG format
        dataset = self.source_dataset if network == 'source' else self.target_dataset
        data, idx2id = networkx_to_pyg(dataset)

        # DEBUG
        print(f"data: {data}")

        # Split both networks in train, val and test
        transform = RandomLinkSplit(
            num_val=0.2,
            num_test=0.0,
            disjoint_train_ratio=0.3,
            add_negative_train_samples=False,   # We will add them in the SAGE model
            is_undirected=True,
        )

        train_data, val_data, test_data = transform(data)

        # DEBUG
        print("Unique train labels:", torch.unique(train_data.edge_label))
        print("Unique val labels:", torch.unique(val_data.edge_label))
        print("Unique test labels:", torch.unique(test_data.edge_label))

        # Get data loaders. Here we sample the subgraphs using the 
        # link neighbors.
        train_loader = LinkNeighborLoader(
            data=train_data,
            subgraph_type="bidirectional",
            num_neighbors=[10, 10],
            neg_sampling="binary",
            num_workers=6,
            edge_label_index=train_data.edge_label_index,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = LinkNeighborLoader(
            data=val_data,
            subgraph_type="bidirectional",
            num_neighbors=[10, 10],
            edge_label_index=val_data.edge_label_index,
            edge_label=val_data.edge_label,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Define models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        embedder = get_embedding_model(
            model=self.embedding,
            in_channels=train_data.num_features,
            hidden_channels=self.hidden_size,
            num_layers=self.num_layers,
            out_channels=self.output_size,
            dropout=0.2
        ).to(device)

        predictor = get_prediction_model(self.prediction, input_dim=self.output_size).to(device)

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(list(embedder.parameters()) + list(predictor.parameters()), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        # ====================
        #   TRAIN & VALIDATE
        # ====================
        best_val_loss = float('inf')
        patience = 10  # Number of epochs to wait for improvement
        patience_counter = 0
        best_model_state_dict = None

        for epoch in range(self.epochs):
            print(f"=== Epoch {epoch} ===")

            # TRAINING
            embedder.train()
            predictor.train()

            train_loss = 0.0
            train_examples = 0

            # Mini-batching
            for batch in tqdm.tqdm(train_loader, desc="Training: "):
                
                batch = batch.to(device)
                optimizer.zero_grad()

                # DEBUG
                # print(batch)
                # print(f"x dtype: {batch.x.dtype}")
                # print(f"edgel_label_index dtype: {batch.edge_label_index.dtype}")
                # print(f"edgel_index dtype: {batch.edge_index.dtype}")
                
                # Get the embeddings
                # (TODO: Concat with hypervectors)
                h = embedder(batch.x, batch.edge_label_index, edge_attr=batch.edge_attr, num_sampled_edges_per_hop=self.batch_size)
                h_src = h[batch.edge_label_index[0]]
                h_dst = h[batch.edge_label_index[1]]
                
                # Predict links
                link_pred = predictor.pred(h_src, h_dst)

                # Backward step
                loss = criterion(link_pred, batch.edge_label)
                loss.backward()
                optimizer.step()

                train_loss += float(loss) * link_pred.numel()
                train_examples += link_pred.numel()

            print(f"Train Loss: {train_loss / train_examples:.4f}")

            # VALIDATION
            embedder.eval()
            predictor.eval()

            val_loss = 0.0
            val_examples = 0

            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc="Validation: "):
                    
                    batch = batch.to(device)

                    # Get the embeddings
                    # (TODO: Concat with hypervectors)
                    h = embedder(batch.x, batch.edge_label_index, edge_attr=batch.edge_attr, num_sampled_edges_per_hop=self.batch_size)
                    h_src = h[batch.edge_label_index[0]]
                    h_dst = h[batch.edge_label_index[1]]

                    # Predict links
                    link_pred = predictor.pred(h_src, h_dst)

                    loss = criterion(link_pred, batch.edge_label)

                    val_loss += float(loss) * link_pred.numel()
                    val_examples += link_pred.numel()

            val_loss /= val_examples
            print(f"Validation Loss: {val_loss:.4f}", end="\n\n")

            # EARLY STOPPING
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save the best model found till now
                best_model_state_dict = {
                    'embedder': embedder.state_dict(),
                    'predictor': predictor.state_dict(),
                }

                patience_counter = 0

            else:
                patience_counter += 1
                print(f"*** Patience counter: {patience_counter} ***")

            if patience_counter >= patience:
                print("*** Early stopping triggered!***", end="\n")
                break
        
        # Load best state dicts
        embedder.load_state_dict(best_model_state_dict['embedder'])
        predictor.load_state_dict(best_model_state_dict['predictor'])

        # Get the node embeddings for the input network
        # using the best embedding model.
        embedder.eval()
        data = data.to(device)
        with torch.no_grad():
            final_h = embedder(data.x, data.edge_index)

        return final_h
        """
