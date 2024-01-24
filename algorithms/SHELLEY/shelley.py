import torch
import torch.optim as optim
from easydict import EasyDict as edict

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SHELLEY.head import get_head
from algorithms.SHELLEY.utils.pyg_convertion import networkx_to_pyg
from utils.graph_utils import load_gt


class SHELLEY(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_dict = args.train_dict
        self.gt = torch.from_numpy(load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx))
        self.use_pretrained = args.use_pretrained
        self.S = None

        # setup model configuration
        self.cfg = edict()

        # backbone
        self.cfg.backbone = edict()
        self.cfg.backbone.name = args.backbone

        if args.backbone == 'gin':
            self.cfg.backbone.node_feature_dim = args.node_feature_dim
            self.cfg.backbone.dim = args.dim
            self.cfg.backbone.out_channels = args.dim
            # self.cfg.backbone.miss_match_value = args.miss_match_value
        else:
            raise Exception(f"Invalid backbone: {self.backbone}.")

        # head
        self.cfg.model = edict()
        self.cfg.model.name = args.head

        if args.head == 'sigma':
            pass # TODO
        elif args.head == 'common':
            self.cfg.model.momentum = args.distill_momentum
            self.cfg.model.warmup_step = args.warmup_step
            self.cfg.model.epoch_iters = args.epoch_iters
            self.cfg.model.alpha = args.alpha
            self.cfg.model.distill = args.distill
        else:
            raise Exception(f"Invalid head: {self.head}.")
        
        # device
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda is True) else 'cpu')

    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first.")
        return self.S
    
    def align(self):
        
        if self.use_pretrained:
            raise Exception("Not implemented yet.") # TODO
        
        # init head
        self.model = get_head(name=self.cfg.model.name, cfg=self.cfg).to(self.device)

        # get the inputs for the model
        self.source_graph = networkx_to_pyg(self.source_dataset.G,
                                            gen_node_feats=True,
                                            ref_node_metric='degree').to(self.device)
        self.target_graph = networkx_to_pyg(self.target_dataset.G,
                                            gen_node_feats=True,
                                            ref_node_metric='degree').to(self.device)
        
        # DEBUG: add feat importance
        self.source_graph.x_importance = torch.ones([self.source_graph.num_nodes, 1]).to(self.device)
        self.target_graph.x_importance = torch.ones([self.target_graph.num_nodes, 1]).to(self.device)
        print('source_graph:', self.source_graph)
        print('target_graph:', self.target_graph)
        

        # train model
        if self.train:
            self.model = self.train_eval_model()
        
        # get final alignment
        inputs = edict()
        inputs.source_graph = self.source_graph
        inputs.target_graph = self.target_graph
        inputs.groundtruth = self.gt

        self.model.eval()
        outputs = self.model(inputs, training=False)
        self.S = outputs.perm_mat.detach().cpu().numpy()

        return self.S


    def train_eval_model(self):
        """
        Train the model.
        """

        # -----------------
        # configue training
        # -----------------
        model_params = self.model.parameters()

        # get optimizer
        if self.cfg.train.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=self.cfg.train.lr, momentum=self.cfg.train.momentum, nesterov=True)
        if self.cfg.train.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=self.cfg.train.lr)
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train.optimizer}.")