import argparse

import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.transforms import RandomLinkSplit

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SANE.prediction_models import get_prediction_model
from algorithms.SANE.sane_utils import networkx_to_pyg
from input.dataset import Dataset


class SANE(NetworkAlignmentModel):
    def __init__(self, source_dataset=None, target_dataset=None, prediction='cosine_similarity', hidden_size=64, num_layers=2, output_size=64, epochs=100, lr=1e-3, batch_size=256):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.prediction=prediction
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    
    def align(self):
        embedder_source = self.train_unsup(network='source')
        embedder_target = self.train_unsup(network='target')
        return 0

    
    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix


    def train_unsup(self, network='source'):
        # Convert the NetworkX to PyG format
        dataset = self.source_dataset if network == 'source' else self.target_dataset
        data, idx2id = networkx_to_pyg(dataset)

        # DEBUG
        print(f"data: {data}")

        # Split both networks in train, val and test
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
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

        embedder = GraphSAGE(
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
                h = embedder(batch.x, batch.edge_label_index, edge_attr=batch.edge_attr)#, num_sampled_edges_per_hop=self.batch_size)
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
                    h = embedder(batch.x, batch.edge_label_index)
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
        return embedder, predictor
    

def parse_args():
    parser = argparse.ArgumentParser(description="SANE")
    parser.add_argument('--prefix1', default="dataspace/douban/online/graphsage")
    parser.add_argument('--prefix2', default="dataspace/douban/offline/graphsage")
    parser.add_argument('--prediction', default="cosine_similarity")
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--groundtruth', default=None)

    return parser.parse_args()


def main(args):

    model = SANE(
        source_dataset=Dataset(args.prefix1),
        target_dataset=Dataset(args.prefix2),
        prediction=args.prediction,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    S = model.align()

    print(S)

if __name__ == "__main__":
    args = parse_args()
    main(args)
