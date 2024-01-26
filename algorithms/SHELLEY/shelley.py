import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict as edict

import evaluation.metrics as metrics
from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SHELLEY.model.head import Common
from algorithms.SHELLEY.model.backbone import GIN
from algorithms.SHELLEY.utils.pyg_convertion import networkx_to_pyg
from algorithms.SHELLEY.utils.split_dict import shuffle_and_split
from algorithms.SHELLEY.model.loss import get_loss_function
from utils.graph_utils import load_gt


def assemble_model(head: str, backbone: str, cfg: dict, device: torch.device = 'cpu'):
    """
    Here the Frankenstein monster comes to life.

    Args:
        head:       The logic of the monster:
                    how to align the node features.
        
        backbone:   The body of the monster:
                    how to generate the node features.
        
        device:     The city in which the monster will live:
                    CPU or GPU?
    
    Returns:
        model:      The Frankenstein monster, ready to align 
                    everything that comes to hand.
    """

    # init backbone
    if backbone == 'gin':
        backbone_model = GIN(in_channels=cfg.BACKBONE.IN_CHANNELS,
                             out_channels=cfg.BACKBONE.OUT_CHANNELS,
                             dim=cfg.BACKBONE.DIM,
                             bias=True)
    else:
        raise Exception(f"[SHELLEY] Invalid backbone: {backbone}")
    
    # init head
    if head == 'common':
        model = Common(backbone=backbone_model,
                            cfg=cfg)
    else:
        raise Exception(f"[SHELLEY] Invalid head: {head}.")
    

    return model


    

class SHELLEY(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_eval_dict = args.train_dict
        self.use_pretrained = args.use_pretrained
        self.skip_train = args.skip_train
        self.eval = args.eval
        self.seed = args.seed
        self.gt_mat = torch.from_numpy(load_gt(self.train_eval_dict, source_dataset.id2idx, target_dataset.id2idx, 'matrix'))
        self.S = None

        if args.eval:
            # split the training dictionary in half and use the
            # first half for training and second for evalutation 
            gt_dict = load_gt(self.train_eval_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
            train_dict, eval_dict = shuffle_and_split(gt_dict, seed=self.seed)
            
            # train:
            self.gt_train_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k,v in train_dict.items()}
            self.source_train_nodes = np.array(list(self.gt_train_dict.keys()))

            # eval:
            self.gt_eval_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k,v in eval_dict.items()}
            self.source_eval_nodes = np.array(list(self.gt_eval_dict.keys()))
        else:
            # use all `train_eval_dict` for training
            gt_train = load_gt(self.train_eval_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
            self.gt_train_mat = torch.from_numpy(load_gt(self.train_eval_dict, source_dataset.id2idx, target_dataset.id2idx))
            self.gt_train_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k,v in gt_train.items()}
            self.source_train_nodes = np.array(list(self.gt_train_dict.keys()))

        # set device
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda is True) else 'cpu')

        # models configuration
        self.cfg = edict()

        # backbone
        self.cfg.backbone = edict()
        self.cfg.backbone.name = args.backbone

        if args.backbone == 'gin':
            self.cfg.backbone.node_feature_dim = args.node_feature_dim
            self.cfg.backbone.dim = args.dim
            self.cfg.backbone.out_channels = args.dim
            self.cfg.backbone.softmax_temp = args.softmax_temp
            # self.cfg.backbone.miss_match_value = args.miss_match_value
        else:
            raise Exception(f"Invalid backbone: {self.backbone}.")
        
        # backbone training
        # if args.train_backbone:
        #     self.cfg.backbone.train = edict()
        #     self.cfg.backbone.train.optimizer = args.backbone_optimizer
        #     self.cfg.backbone.train.lr = args.backbone_lr
        #     self.cfg.backbone.train.use_scheduler = args.backbone_use_scheduler
        #     self.cfg.backbone.train.lr_step = args.backbone_lr_step 
        #     self.cfg.backbone.train.lr_decay = args.backbone_lr_decay
        #     self.cfg.backbone.train.start_epoch = args.backbone_start_epoch
        #     self.cfg.backbone.train.num_epochs = args.backbone_num_epochs
        #     self.cfg.backbone.train.batchsize = args.backbone_batchsize # do not perform mini-batching if `args.batchsize` == -1
        #     self.cfg.backbone.train.loss = args.backbone_loss
        
        # head
        self.cfg.MODEL = edict()
        self.cfg.MODEL.NAME = args.head

        if args.head == 'sigma':
            pass # TODO
        elif args.head == 'common':
            self.cfg.HEAD.DISTILL = args.distill
            self.cfg.HEAD.MOMENTUM = args.distill_momentum
            # retrieve warmup_step from the `batchsize` argument if `args.warmup_step` == -1
            self.cfg.HEAD.WARMUP_STEP = len(self.source_train_nodes) // args.batchsize if args.warmup_step == -1 else args.warmup_step 
            self.cfg.HEAD.EPOCH_ITERS = args.epoch_iters
            self.cfg.HEAD.ALPHA = args.alpha
        elif args.head == 'stablegm':
            self.cfg.HEAD.FEATURE_CHANNEL = args.feature_channel
            self.cfg.HEAD.SK_ITER_NUM = args.sk_iter_num
            self.cfg.HEAD.SK_EPSILON = args.sk_epsilon
            self.cfg.HEAD.SK_TAU = args.sk_tau
        else:
            raise Exception(f"Invalid head: {self.head}.")
        
        # head training
        self.cfg.TRAIN = edict()
        self.cfg.TRAIN.OPTIMIZER = args.optimizer
        self.cfg.TRAIN.MOMENTUM = args.optim_momentum 
        self.cfg.TRAIN.LR = args.lr
        self.cfg.TRAIN.USE_SCHEDULER = args.use_scheduler
        self.cfg.TRAIN.LR_STEP = args.lr_step 
        self.cfg.TRAIN.LR_DECAY = args.lr_decay
        self.cfg.TRAIN.START_EPOCH = args.start_epoch
        self.cfg.TRAIN.NUM_EPOCHS = args.num_epochs
        self.cfg.TRAIN.BATCH_SIZE = len(self.source_train_nodes) if args.batchsize == -1 else args.batchsize # do not perform mini-batching if `args.batchsize` == -1
        self.cfg.TRAIN.LOSS = args.loss

    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first.")
        return self.S
    
    def align(self):
        
        if self.use_pretrained:
            raise Exception("Not implemented yet.") # TODO
        
        # reproducibility
        torch.random.manual_seed(self.seed)     # --------
        torch.manual_seed(self.seed)            #   NOTE
        np.random.seed(self.seed)               # --------

        # init model
        self.model = assemble_model(head=self.cfg.HEAD.NAME, 
                                    backbone=self.cfg.BACKBONE.NAME,
                                    device=self.device)

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
        if not self.skip_train:
            self.train_eval_model()
        
        # get final alignment
        self.model.eval()

        inputs = edict()
        inputs.pyg_graphs = [self.source_graph, self.target_graph]
        inputs.ns = [self.source_graph.num_nodes, self.target_graph.num_nodes]

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

        # loss function
        criterion = get_loss_function(self.cfg.model.train.loss)
        
        # optimizer
        model_params = self.model.parameters()

        if self.cfg.model.train.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=self.cfg.model.train.lr, momentum=self.cfg.model.train.momentum, nesterov=True)
        elif self.cfg.model.train.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=self.cfg.model.train.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.model.train.optimizer}.")
        
        # scheduler
        if self.cfg.model.train.use_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.model.train.lr_step,
                gamma=self.cfg.model.train.lr_decay,
                last_epoch=self.cfg.model.train.start_epoch - 1
            )
        
        # ---------------
        # train/eval loop
        # ---------------
        start_epoch = self.cfg.model.train.start_epoch
        num_epochs = self.cfg.model.train.num_epochs
        batch_size = self.cfg.model.train.batchsize
        mb_iters = len(self.source_train_nodes) // batch_size
        assert mb_iters > 0, "batch_size is too large"
        if(len(self.source_train_nodes) % batch_size > 0):
            mb_iters += 1

        for epoch in range(start_epoch, num_epochs):
            # reset seed
            # torch.manual_seed(self.seed + epoch + 1)

            print('\n')
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

            # epoch_loss = 0.0              # --------
            # running_loss = 0.0            #   TODO
            # running_since = time.time()   #   DOTO
            # iter_num = 0                  # --------

            # training:
            self.model.train()  # set model to train mode
            np.random.shuffle(self.source_train_nodes)
            for iter in range(mb_iters):
                # get batch inputs
                source_batch = self.source_train_nodes[iter*batch_size:(iter+1)*batch_size]
                target_batch = [self.gt_train_dict[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch).to(self.device)
                target_batch = torch.LongTensor(target_batch).to(self.device)

                inputs = edict()
                inputs.pyg_graphs = [self.source_graph, self.target_graph]
                inputs.ns = [self.source_graph.num_nodes, self.source_graph.num_nodes]
                inputs.gt_perm_mat = [self.gt_train_mat]

                # zero gradient params
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward step
                    if 'common' in self.cfg.model.name:
                        inputs = edict()
                        inputs.source_graph = self.source_graph.to(self.device)
                        inputs.target_graph = self.target_graph.to(self.device)
                        inputs.source_batch = source_batch
                        inputs.target_batch = target_batch
                        inputs.gt_perm_mat = self.gt_mat.to(self.device)[source_batch, :][:, target_batch]
                        
                        outputs = self.model(inputs, training=True, iter_num=iter, epoch=epoch)
                    elif 'stablegm' in self.cfg.model.name:
                        outputs = self.model(inputs)
                    else:
                        raise Exception(f"Invalid model head {self.cfg.model.name}")
                    
                    # check output
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs

                    # compute loss
                    if self.cfg.model.train.loss == 'cml':
                        loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'],outputs['perm_mat'], *outputs['ns'], beta=0.1)
                    elif self.cfg.model.train.loss == 'common':
                        loss = torch.sum(outputs.loss)
                    else:
                        raise Exception(f"Invalid loss function: {self.cfg.model.train.loss}")
                    
                    # backward step
                    loss.backward()
                    optimizer.step()

                    print(f"train loss: {loss}")

            # evalutation:
            if self.eval:
                self.model.eval()   # set model to eval mode
                # np.random.shuffle(self.source_eval_nodes)   # NOTE: maybe not?
                with torch.no_grad():
                    source_batch = self.source_eval_nodes
                    target_batch = [self.gt_eval_dict[x] for x in source_batch]
                    source_batch = torch.LongTensor(source_batch).to(self.device)
                    target_batch = torch.LongTensor(target_batch).to(self.device)
                    
                    if 'common' in self.cfg.model.name:
                        inputs = edict()
                        inputs.source_graph = self.source_graph.to(self.device)
                        inputs.target_graph = self.target_graph.to(self.device)
                        inputs.source_batch = source_batch
                        inputs.target_batch = target_batch
                        inputs.gt_perm_mat = self.gt_mat.to(self.device)[source_batch, :][:, target_batch]

                        outputs = self.model(inputs, training=False)

                        # compute matching accuracy on eval subset
                        pred = outputs.perm_mat[source_batch, :][:, target_batch]   # extract only rows and columns corresponding to the batch
                        # pred = metrics.greedy_match(pred.detach().cpu().numpy())
                        pred = pred.detach().cpu().numpy()
                        match_acc = metrics.compute_accuracy(pred, outputs.gt_perm_mat)
                        print(f'eval matching accuracy: {match_acc:.4f}')

                    else:
                        raise Exception(f"Invalid model head {self.cfg.model.name}")
                    

            # update scheduler
            if self.cfg.model.train.use_scheduler:
                scheduler.step()