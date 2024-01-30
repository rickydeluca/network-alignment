import time

import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SHELLEY.model.backbone import GIN
from algorithms.SHELLEY.model.head import Common, StableGM
from algorithms.SHELLEY.model.loss import ContrastiveLossWithAttention
from algorithms.SHELLEY.utils.evaluation_metric import matching_accuracy
from algorithms.SHELLEY.utils.networkx_to_pyg import networkx_to_pyg
from algorithms.SHELLEY.utils.split_dict import shuffle_and_split

# from algorithms.SHELLEY.utils.data_to_cuda import data_to_cuda
from utils.graph_utils import load_gt


def assemble_model(head: str, backbone: str, cfg: dict, device: torch.device = 'cpu'):
    """
    Here the Frankenstein's Monster comes to life.

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
        backbone_model = GIN(in_channels=cfg.BACKBONE.NODE_FEATURE_DIM,
                             out_channels=cfg.BACKBONE.OUT_CHANNELS,
                             dim=cfg.BACKBONE.DIM,
                             bias=True).to(device)
    else:
        raise Exception(f"[SHELLEY] Invalid backbone: {backbone}")
    
    # init head
    if head == 'common':
        model = Common(backbone=backbone_model, cfg=cfg).to(device)
    elif head == 'stablegm':
        model = StableGM(backbone=backbone_model, cfg=cfg).to(device)
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
        self.cfg.BACKBONE = edict()
        self.cfg.BACKBONE.NAME = args.backbone

        if args.backbone == 'gin':
            self.cfg.BACKBONE.NODE_FEATURE_DIM = args.node_feature_dim
            self.cfg.BACKBONE.DIM = args.dim
            self.cfg.BACKBONE.OUT_CHANNELS = args.dim
            # self.cfg.backbone.miss_match_value = args.miss_match_value
        else:
            raise Exception(f"Invalid backbone: {self.backbone}.")
        
        # # backbone training
        # if args.train_backbone:
        #     self.cfg.BACKBONE.TRAIN = edict()
        #     self.cfg.BACKBONE.TRAIN.OPTIMIZER = args.backbone_optimizer
        #     self.cfg.BACKBONE.TRAIN.LR = args.backbone_lr
        #     self.cfg.BACKBONE.TRAIN.USE_SCHEDULER = args.backbone_use_scheduler
        #     self.cfg.BACKBONE.TRAIN.LR_STEP = args.backbone_lr_step 
        #     self.cfg.BACKBONE.TRAIN.LR_DECAY = args.backbone_lr_decay
        #     self.cfg.BACKBONE.TRAIN.START_EPOCH = args.backbone_start_epoch
        #     self.cfg.BACKBONE.TRAIN.NUM_EPOCHS = args.backbone_num_epochs
        #     self.cfg.BACKBONE.TRAIN.BATCH_SIZE = args.backbone_batchsize # do not perform mini-batching if `args.batchsize` == -1
        #     self.cfg.BACKBONE.TRAIN.LOSS = args.backbone_loss
        
        # head
        self.cfg.HEAD = edict()
        self.cfg.HEAD.NAME = args.head

        if args.head == 'sigma':
            pass # TODO
        elif args.head == 'common':
            self.cfg.HEAD.DISTILL = args.distill
            self.cfg.HEAD.MOMENTUM = args.distill_momentum
            # retrieve warmup_step from the `batchsize` argument if `args.warmup_step` == -1
            self.cfg.HEAD.WARMUP_STEP = len(self.source_train_nodes) // args.batchsize if args.warmup_step == -1 else args.warmup_step 
            self.cfg.HEAD.EPOCH_ITERS = args.epoch_iters
            self.cfg.HEAD.ALPHA = args.alpha
            self.cfg.HEAD.SOFTMAX_TEMP = args.softmax_temp
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
        self.cfg.TRAIN.SEPARATE_BACKBONE_LR = args.separate_backbone_lr
        self.cfg.TRAIN.BACKBONE_LR = args.backbone_lr
        self.cfg.TRAIN.START_EPOCH = args.start_epoch
        self.cfg.TRAIN.NUM_EPOCHS = args.num_epochs
        self.cfg.TRAIN.EPOCH_ITERS = args.epoch_iters
        self.cfg.TRAIN.BATCH_SIZE = len(self.source_train_nodes) if args.batchsize == -1 else args.batchsize # do not perform mini-batching if `args.batchsize` == -1
        self.cfg.TRAIN.LOSS_FUNC = args.loss_func
        self.cfg.TRAIN.STATISTIC_STEP = args.statistic_step 

    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first.")
        return self.S
    
    def align(self):
        
        if self.use_pretrained:
            raise Exception("Not implemented yet.") # TODO
        
        # reproducibility               # NOTE
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # init model
        self.model = assemble_model(head=self.cfg.HEAD.NAME, 
                                    backbone=self.cfg.BACKBONE.NAME,
                                    cfg=self.cfg,
                                    device=self.device)

        # get the inputs for the model
        self.source_graph = networkx_to_pyg(self.source_dataset.G,
                                            gen_node_feats=True,
                                            ref_node_metric='degree').to(self.device)
        self.target_graph = networkx_to_pyg(self.target_dataset.G,
                                            gen_node_feats=True,
                                            ref_node_metric='degree').to(self.device)
        
        # set feature importance
        self.source_graph.x_importance = torch.ones([self.source_graph.num_nodes, 1]).to(self.device)
        self.target_graph.x_importance = torch.ones([self.target_graph.num_nodes, 1]).to(self.device)

        # TODO: define dataloader
        self.dataloader = DataLoader()

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

        # ------------------
        # Configure training
        # ------------------

        # loss function
        if self.cfg.TRAIN.LOSS_FUNC.lower() == 'cml':
            criterion = ContrastiveLossWithAttention()
        elif self.cfg.TRAIN.LOSS_FUNC.lower() == 'distill_qc':
            print("[SHELLEY] You selected 'distill_quadratic' as loss function which is defined within the model,"
                  "please ensure there is a tensor with key 'loss' in your model\'s returned dictionary.")
            criterion = None
        else:
            raise Exception(f"[SHELLEY] Invalid loss function: {self.cfg.TRAIN.LOSS_FUNC.lower()}.")
        
        # get model parameters
        if self.cfg.TRAIN.SEPARATE_BACKBONE_LR:
            backbone_ids = [id(item) for item in self.model.backbone_params]
            other_params = [param for param in self.model.parameters() if id(param) not in backbone_ids]

            model_params = [
                {'params': other_params},
                {'params': self.model.backbone_params, 'lr': self.cfg.TRAIN.BACKBONE_LR}
            ]

        else:
            model_params = self.model.parameters()
        
        # optimizer
        if self.cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=self.cfg.TRAIN.LR, momentum=self.cfg.TRAIN.MOMENTUM, nesterov=True)
        elif self.cfg.TRAIN.OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=self.cfg.TRAIN.LR)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.TRAIN.OPTIMIZER}.")
        
        # scheduler
        if self.cfg.TRAIN.USE_SCHEDULER:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.TRAIN.LR_STEP,
                gamma=self.cfg.TRAIN.LR_DECAY,
                last_epoch=self.cfg.TRAIN.START_EPOCH - 1
            )
        
        # ---------------
        # Train/Eval loop
        # ---------------
        print("Start training...")
        since = time.time()
        start_epoch = self.cfg.TRAIN.START_EPOCH
        num_epochs = self.cfg.TRAIN.NUM_EPOCHS

        for epoch in range(start_epoch, num_epochs):
            # reset seed after evaluation per epoch
            torch.manual_seed(self.seed + epoch + 1)

            # TODO: get dataloader

            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # === TRAINING ===

            # set model to training mode
            self.model.train()
            # self.model.module.trainings = True  # NOTE

            print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

            epoch_loss = 0.0
            epoch_acc = 0.0
            running_loss = 0.0
            running_acc = 0.0
            running_since = time.time()
            iter_num = 0

            # iterate over data
            for inputs in self.dataloader['train']:

                if iter_num >= self.cfg.TRAIN.EPOCH_ITERS:
                    break
                
                # TODO: move data to correct device
                # if self.model.device != torch.device('cpu'):
                #     inputs.pyg_graphs = [g.to(self.device) for g in inputs.pyg_graphs]

                iter_num += 1

                # zero the gradient parameters
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward step
                    if 'common' in self.cfg.HEAD.NAME: # COMMON use the iter number to control the warmup temperature
                        outputs = self.model(inputs, training=True, iter_num=iter_num, epoch=epoch)
                    else:
                        outputs = self.model(inputs)

                    # compute loss
                    if self.cfg.TRAIN.LOSS_FUNC == 'cml':
                        loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], outputs['perm_mat'], *outputs['ns'], beta=0.1)
                    elif self.cfg.TRAIN.LOSS_FUNC == 'distill_qc':
                        loss = torch.sum(outputs['loss'])
                    else:
                        raise ValueError(f"[SHELLEY] Unsupperted loss function: {self.cfg.TRAIN.LOSS_FUNC}")

                    # compute accuracy
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)[0]

                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                    batch_num = inputs['batch_size']

                    # statistics
                    running_loss += loss.item() * batch_num
                    running_acc += acc.item() * batch_num
                    epoch_loss += loss.item() * batch_num
                    epoch_acc += acc.item() * batch_num
                    

                    if iter_num % self.cfg.TRAIN.STATISTIC_STEP == 0:
                        running_speed = self.cfg.TRAIN.STATISTIC_STEP * batch_num / (time.time() - running_since)
                        print(f'Epoch: {epoch:<4}' \
                              f'Iteration: {iter_num:<4}' \
                              f'{running_speed:>4.2f}sample/s'\
                              f'Loss: {running_loss / self.cfg.TRAIN.STATISTIC_STEP / batch_num:<8.4f}'\
                              f'Accuracy: {running_acc / self.cfg.TRAIN.STATISTIC_STEP / batch_num:<8.4f}')

                        running_loss = 0.0
                        running_acc = 0.0
                        running_since = time.time()
        
            # epoch statistics
            epoch_loss = epoch_loss / self.cfg.TRAIN.EPOCH_ITERS / batch_num
            epoch_acc = epoch_acc / self.cfg.TRAIN.EPOCH_ITERS / batch_num
            print(f"[TRAINING] Epoch {epoch+1:<4} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

            # === EVALUATION ===
            
            if self.eval:
                # TODO
                raise Exception("[SHELLEY] Evaluation is not implemented yet!")
                
            # update scheduler
            if self.cfg.TRAIN.USE_SCHEDULER:
                scheduler.step()
        
        # compute elapsed training time
        time_elapsed = time.time() - since 
        hours = time_elapsed // 3600
        minutes = (time_elapsed // 60) % 60
        seconds = time_elapsed % 60
        print(f"Training complete in {hours:.0f}h {minutes:0f}m {seconds:0f}s")

        # TODO: load best model parameters

        return