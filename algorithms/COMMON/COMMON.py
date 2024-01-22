import os
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from algorithms.COMMON.embedding_model import CommonEmbedding
from algorithms.COMMON.mapping_model import CommonMapping
from algorithms.network_alignment_model import NetworkAlignmentModel
from utils.graph_utils import load_gt


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def generate_unique_edge_attr(num_edges):
    edge_attr = torch.rand((num_edges, 2), dtype=torch.float)
    return edge_attr


def dataset_to_pyg(dataset, pos_info=False, normalize=False):
    """
    Given a `Dataset` object, return the corresponding pyg graph.
    """
    # Convert NetworX to Data representation
    G = from_networkx(dataset.G)

    # Extract useful informations
    edge_index = G.edge_index
    edge_weight = G.weight if "weight" in G.keys() else None

    # Read node/edge features
    x = (
        torch.tensor(dataset.features, dtype=torch.float32)
        if dataset.features is not None
        else None
    )
    edge_attr = (
        torch.tensor(dataset.edge_features)
        if dataset.edge_features is not None
        else None
    )

    # Use positional informations if required
    if pos_info:
        adj_matrix = torch.tensor(
            nx.adjacency_matrix(dataset.G).todense(), dtype=torch.float32
        )
        if x is not None:
            x = torch.cat((x, adj_matrix), dim=1)

    # Normalize attributes if required
    if normalize:
        x = normalize_over_channels(x)
        edge_attr = normalize_over_channels(edge_attr)

    # Build the PyG Data object
    data = Data(
        x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr
    )

    return data


def networkx_to_pyg(G, node_feats=None, edge_feats=None, normalize=False):
    pyg_graph = from_networkx(G)

    if node_feats is not None:
        # x = torch.from_numpy(node_feats)
        x = node_feats
        if normalize:
            x = normalize_over_channels(x)
    else:
        x = None

    if edge_feats is not None:
        # edge_attr = torch.from_numpy(edge_feats)
        edge_attr = edge_feats
        if normalize:
            edge_attr = normalize_over_channels(edge_attr)
    else:
        edge_attr = generate_unique_edge_attr(pyg_graph.num_edges)

    # DEBUG
    print('edge_attr shape:', edge_attr.shape)

    pyg_graph.x = x
    pyg_graph.edge_attr = edge_attr

    return pyg_graph


class COMMON(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        super().__init__(source_dataset, target_dataset)

        # Default parameters
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_dict = args.train_dict

        self.gt = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, "dict")
        self.gt_train = {
            self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v]
            for k, v in self.gt.items()
        }
        self.gt_train_perm_mat = torch.from_numpy(load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx))

        self.seed = args.seed
        self.cuda = args.cuda
        self.device = torch.device(
            "cuda:0" if (self.cuda and torch.cuda.is_available()) else "cpu"
        )

        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_train_nodes = np.array(list(self.gt_train.keys()))

        # Encoding model parameters
        self.emb_batch_size = args.emb_batch_size
        self.emb_epochs = args.emb_epochs
        self.emb_lr = args.emb_lr
        self.neg_sample_size = args.neg_sample_size
        self.embedding_dim = args.embedding_dim
        self.embedding_name = args.embedding_name

        # Mapping model parameters
        self.map_batch_size = args.map_batch_size
        self.map_epochs = args.map_epochs
        self.map_epoch_iters = args.map_epoch_iters
        self.map_lr = args.map_lr
        self.map_lr_decay = args.map_lr_decay
        self.map_lr_step = args.map_lr_step
        self.map_optimizer = args.map_optimizer
        self.map_loss_func = args.map_loss_func
        self.backbone = args.backbone
        self.rescale = args.rescale
        self.separate_backbone_lr = args.separate_backbone_lr
        self.backbone_lr = args.backbone_lr
        self.map_optimizer_momentum = args.map_optimizer_momentum
        self.map_softmax_temp = args.map_softmax_temp
        self.map_eval_epochs = args.map_eval_epochs
        self.map_feature_channel = args.map_feature_channel
        self.alpha = args.alpha
        self.distill = args.distill
        self.warmup_step = args.warmup_step
        self.distill_momentum = args.distill_momentum

        # Reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception(
                "Must calculate alignment matrix by calling 'align()' method first!"
            )
        return self.S

    def align(self):
        # Learn starting embeddings using the same module from PALE
        self.learn_embeddings()

        # Save learned embeddings in `word2vec` format
        self.to_word2vec_format(
            self.source_embedding,
            self.source_dataset.G.nodes(),
            "algorithms/COMMON/embeddings",
            self.embedding_name + "_source",
            self.embedding_dim,
            self.source_dataset.id2idx,
        )

        self.to_word2vec_format(
            self.target_embedding,
            self.target_dataset.G.nodes(),
            "algorithms/COMMON/embeddings",
            self.embedding_name + "_target",
            self.embedding_dim,
            self.target_dataset.id2idx,
        )

        # Learn the node alignments
        self.learn_alignment()

        # Compute alignment with all the dataset
        self.mapping_model.eval()
        self.mapping_model.moudle.trainings = False

        inputs = {'source_graph': self.source_graph,
                  'target_graph': self.target_graph}
        outputs = self.mapping_model(inputs)

        self.S = outputs['perm_mat']
        return self.S

    # ======================
    #   EMBEDDING LEARNING
    # ======================
    def learn_embeddings(self):
        num_source_nodes = len(self.source_dataset.G.nodes())
        source_deg = self.source_dataset.get_nodes_degrees()
        source_edges = self.source_dataset.get_edges()

        num_target_nodes = len(self.target_dataset.G.nodes())
        target_deg = self.target_dataset.get_nodes_degrees()
        target_edges = self.target_dataset.get_edges()

        # source_edges, target_edges = self.extend_edge(source_edges, target_edges)

        print("Done extend edges")
        self.source_embedding = self.learn_embedding(
            num_source_nodes, source_deg, source_edges
        )
        self.target_embedding = self.learn_embedding(
            num_target_nodes, target_deg, target_edges
        )

    def learn_embedding(self, num_nodes, deg, edges):
        embedding_model = CommonEmbedding(
            n_nodes=num_nodes,
            embedding_dim=self.embedding_dim,
            deg=deg,
            neg_sample_size=self.neg_sample_size,
            cuda=self.cuda,
        )

        if self.cuda:
            embedding_model = embedding_model.cuda()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, embedding_model.parameters()),
            lr=self.emb_lr,
        )
        embedding = self.train_embedding(embedding_model, edges, optimizer)

        return embedding

    def train_embedding(self, embedding_model, edges, optimizer):
        n_iters = len(edges) // self.emb_batch_size
        assert n_iters > 0, "batch_size is too large!"
        if len(edges) % self.emb_batch_size > 0:
            n_iters += 1
        print_every = int(n_iters / 4) + 1
        total_steps = 0
        n_epochs = self.emb_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time.time()

            print("Epoch {0}".format(epoch))
            np.random.shuffle(edges)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(
                    edges[iter * self.emb_batch_size : (iter + 1) * self.emb_batch_size]
                )
                if self.cuda:
                    batch_edges = batch_edges.cuda()
                start_time = time.time()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(
                    batch_edges[:, 0], batch_edges[:, 1]
                )
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0:
                    print(
                        "Iter:", "%03d" % iter,
                        "train_loss=", "{:.5f}".format(loss.item()),
                        "true_loss=", "{:.5f}".format(loss0.item()),
                        "neg_loss=", "{:.5f}".format(loss1.item()),
                        "time", "{:.5f}".format(time.time() - start_time),
                    )
                total_steps += 1

            # for time evaluate
            self.embedding_epoch_time = time.time() - start

        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.cuda:
            embedding = embedding.cuda()

        return embedding

    def to_word2vec_format(
        self, val_embeddings, nodes, out_dir, filename, dim, id2idx, pref=""
    ):
        val_embeddings = val_embeddings.cpu().detach().numpy()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open("{0}/{1}".format(out_dir, filename), "w") as f_out:
            f_out.write("%s %s\n" % (len(nodes), dim))
            for node in nodes:
                txt_vector = [
                    "%s" % val_embeddings[int(id2idx[node])][j] for j in range(dim)
                ]
                f_out.write("%s%s %s\n" % (pref, node, " ".join(txt_vector)))
            f_out.close()
        print("Embedding has been saved to: {0}/{1}".format(out_dir, filename))

    # ======================
    #   ALIGNMENT LEARNING
    # ======================
    def learn_alignment(self):
        # Generate the pytorch geometric graph objects using the learned embeddings
        self.source_graph = networkx_to_pyg(self.source_dataset.G,
                                       node_feats=self.source_embedding,
                                       normalize=True).to(self.device)
        self.target_graph = networkx_to_pyg(self.target_dataset.G,
                                       node_feats=self.target_embedding,
                                       normalize=True).to(self.device)

        # Define model
        model = CommonMapping(
            # source_embedding=self.source_embedding,  # Learned in the...
            # target_embedding=self.target_embedding,  # ... encoding layer
            backbone=self.backbone,
            feature_channel=self.map_feature_channel,
            softmax_temp=self.map_softmax_temp,
            momentum=self.distill_momentum,
            warmup_step=self.warmup_step,
            epoch_iters=self.map_epoch_iters,
            rescale=self.rescale,
            alpha=self.alpha
        )

        model = model.to(self.device)

        # Criterion
        if self.map_loss_func == "custom":
            criterion = None
        else:
            raise ValueError(f"Unknown loss function: {self.map_loss_func}")

        # Learning rate (with different lr for backbone layer)
        if self.separate_backbone_lr:
            backbone_ids = [id(item) for item in model.backbone_params]
            other_params = [
                param for param in model.parameters() if id(param) not in backbone_ids
            ]

            model_params = [
                {"params": other_params},
                {"params": model.backbone_params, "lr": self.backbone_lr},
            ]
        else:
            model_params = model.parameters()

        # Optimizer
        if self.map_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model_params,
                lr=self.map_lr,
                momentum=self.map_optimizer_momentum,
                nesterov=True,
            )
        elif self.map_optimizer == "adam":
            optimizer = torch.optim.Adam(model_params, lr=self.map_lr)
        else:
            raise ValueError(f"Invalid optimizer: {self.map_optimizer}")

        # Training
        model = self.train_eval_alignment(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=self.map_epochs,
            start_epoch=0
        )

        # Save the trained mapping model
        self.mapping_model = model

    def train_eval_alignment(
        self,
        model=None,
        criterion=None,
        optimizer=None,
        dataset=None,
        dataloader=None,
        num_epochs=25,
        start_epoch=0,
    ):
        print(f"Start alignment training on device '{self.device}'...")

        start_training_time = time.time()

        # Define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.map_lr_step,
            gamma=self.map_lr_decay,
            last_epoch=-1,
        )

        # Compute mini-batching parameters
        num_iters = len(self.source_train_nodes) // self.map_batch_size
        assert num_iters > 0, "map_batch_size is too large"
        print_every = int(num_iters/4) + 1
        total_steps = 0

        # Train loop
        for epoch in range(start_epoch, num_epochs):
            start_epoch_time = time.time()     # for time evaluation
            
            # Reset seed after each evaluation
            torch.manual_seed(self.seed + epoch + 1)

            # Se model to training mode
            model.train()
            # model.modules.trainings = True

            print(f"Epoch {epoch+1}/{num_epochs}")
            print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

            # Mini-batching
            np.random.shuffle(self.source_train_nodes)
            for iter in range(num_iters):
                # Get the batch with the index subset
                source_batch = self.source_train_nodes[iter*self.map_batch_size:(iter+1)*self.map_batch_size]
                target_batch = [self.gt_train[x] for x in source_batch]

                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)

                # Get the subset of the groundtruth training matrix using only the 
                # indices in the batch
                gt_batch_perm_mat = self.gt_train_perm_mat

                # Prepare the input dictionary
                inputs = {'source_graph': self.source_graph,
                          'target_graph': self.target_graph,
                          'source_batch': source_batch,
                          'target_batch': target_batch,
                          'gt_perm_mat': gt_batch_perm_mat}

                # Zero the parameter gradients
                optimizer.zero_grad()
                start_batch_time = time.time()

                # Forward step
                outputs = model(inputs, training=True, iter_num=iter+1, epoch=epoch)

                # Check that the ouput dictionary contains all the required values
                assert 'perm_mat' in outputs
                assert 'gt_perm_mat' in outputs
                assert 'loss' in outputs

                # Compute loss
                if self.map_loss_func == 'custom':
                    loss = torch.sum(outputs['loss'])
                else:
                    raise ValueError(f'Unsupported loss function: {self.map_loss_func}')
                
                # Backward step
                loss.backward()
                optimizer.step()

                # Print mini-batch stats
                if total_steps % print_every == 0 and total_steps > 0:
                    print(f'Iter:\t{iter:.3d}',
                          f'train_loss =\t{loss.item():.5f}',
                          f'time:\t{(time.time()-start_batch_time):.5f}')
                total_steps += 1


            # Save average epoch time
            self.avg_map_epoch_time += (time.time() - start_epoch_time) / num_epochs

            # Update scheduler
            scheduler.step()


        time_elapsed = time.time() - start_training_time
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
            .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
        
        print(f'Average training epoch time: {self.avg_map_epoch_time:.3f}')

        return model