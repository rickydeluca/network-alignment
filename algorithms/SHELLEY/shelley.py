import time

import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict as edict
from torch.utils.data import random_split

from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SHELLEY.model.backbone import GIN
from algorithms.SHELLEY.model.head import Common, StableGM
from algorithms.SHELLEY.model.loss import ContrastiveLossWithAttention
from algorithms.SHELLEY.utils.data_to_cuda import data_to_cuda
from algorithms.SHELLEY.utils.dataset.data_loader import (
    GMSemiSyntheticDataset, get_dataloader)
from algorithms.SHELLEY.utils.evaluation_metric import matching_accuracy
from algorithms.SHELLEY.utils.networkx_to_pyg import networkx_to_pyg
from algorithms.SHELLEY.utils.split_dict import shuffle_and_split
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

    # Backbone
    if backbone == 'gin':
        backbone_model = GIN(in_channels=cfg.BACKBONE.IN_CHANNELS,
                             out_channels=cfg.BACKBONE.OUT_CHANNELS,
                             dim=cfg.BACKBONE.DIM,
                             bias=True).to(device)
    # elif backbone == 'pale_linear':
    #     backbone_model = PaleLinear(in_channels=cfg.BACKBONE.IN_CHANNELS,
    #                          out_channels=cfg.BACKBONE.OUT_CHANNELS,
    #                          bias=True).to(device)
    else:
        raise Exception(f"[SHELLEY] Invalid backbone: {backbone}.")
    
    # Head
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
        self.seed = args.seed
        self.gt_mat = torch.from_numpy(
            load_gt(self.train_eval_dict,
                    source_dataset.id2idx,
                    target_dataset.id2idx,
                    'matrix'))
        self.S = None

        if args.eval:
            # split the training dictionary in half and use the
            # first half for training and second for evalutation 
            gt_dict = load_gt(
                self.train_eval_dict,
                source_dataset.id2idx,
                target_dataset.id2idx,
                'dict')
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

        # Set device
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda is True) else 'cpu')

        # Models configurations
        self.cfg = edict()

        # Data
        self.cfg.DATA = edict()
        self.cfg.DATA.ROOT_DIR = args.root_dir 
        self.cfg.DATA.NAME = args.root_dir.split('/')[-1]
        self.cfg.DATA.P_ADD = args.p_add 
        self.cfg.DATA.P_RM = args.p_rm

        # Backbone
        self.cfg.BACKBONE = edict()
        self.cfg.BACKBONE.NAME = args.backbone

        if args.backbone == 'gin':
            self.cfg.BACKBONE.IN_CHANNELS = args.node_feature_dim
            self.cfg.BACKBONE.DIM = args.embedding_dim
            self.cfg.BACKBONE.OUT_CHANNELS = args.embedding_dim
        elif args.backbone == 'pale_linear':
            self.cfg.BACKBONE.IN_CHANNELS = args.node_feature_dim
            self.cfg.BACKBONE.OUT_CHANNELS = args.embedding_dim
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
        
        # Head
        self.cfg.HEAD = edict()
        self.cfg.HEAD.NAME = args.head
        self.cfg.HEAD.DEVICE = self.device
        if args.head == 'sigma':
            pass # TODO
        elif args.head == 'common':
            self.cfg.HEAD.DISTILL = args.distill
            self.cfg.HEAD.MOMENTUM = args.distill_momentum
            self.cfg.HEAD.WARMUP_STEP = args.warmup_step 
            self.cfg.HEAD.EPOCH_ITERS = args.epoch_iters
            self.cfg.HEAD.ALPHA = args.alpha
            self.cfg.HEAD.SOFTMAX_TEMP = args.softmax_temp
            self.cfg.HEAD.FEATURE_CHANNEL = args.feature_channel
        elif args.head == 'stablegm':
            self.cfg.HEAD.FEATURE_CHANNEL = args.feature_channel
            self.cfg.HEAD.SK_ITER_NUM = args.sk_iter_num
            self.cfg.HEAD.SK_EPSILON = args.sk_epsilon
            self.cfg.HEAD.SK_TAU = args.sk_tau
        else:
            raise Exception(f"Invalid head: {self.head}.")
        
        # Training
        self.cfg.TRAIN = edict()
        self.cfg.TRAIN.TRAIN = args.train
        self.cfg.TRAIN.SELF_SUPERVISED = args.self_supervised  # if True do not use the target graphs in the dataset, but generate a random graph 'on the fly'
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
        self.cfg.TRAIN.BATCH_SIZE = args.batchsize # do not perform mini-batching if `args.batchsize` == -1
        self.cfg.TRAIN.LOSS_FUNC = args.loss_func
        self.cfg.TRAIN.STATISTIC_STEP = args.statistic_step 
        self.cfg.TRAIN.EARLY_STOPPING = args.early_stopping 
        self.cfg.TRAIN.PATIENCE = args.patience

        # Evaluation
        self.cfg.EVAL = edict()
        self.cfg.EVAL.TEST = args.test
        self.cfg.EVAL.VALIDATE = args.validate      # Validation during training
        self.cfg.EVAL.VAL_ITERS = args.val_iters
        self.cfg.EVAL.TEST_ITERS = args.test_iters

        # Checkpoints
        self.cfg.CHECKPOINTS = args.checkpoints

    def get_alignment_matrix(self):
        if self.S is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first.")
        return self.S
    
    def align(self):
        
        if self.use_pretrained:
            raise Exception("Not implemented yet.") # TODO
        
        # Reproducibility               # NOTE
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Init model
        self.model = assemble_model(head=self.cfg.HEAD.NAME, 
                                    backbone=self.cfg.BACKBONE.NAME,
                                    cfg=self.cfg,
                                    device=self.device)

        # Get dataset
        full_dataset = GMSemiSyntheticDataset(root_dir=self.cfg.DATA.ROOT_DIR,
                                              p_add=self.cfg.DATA.P_ADD,
                                              p_rm=self.cfg.DATA.P_RM,
                                              self_supervised=self.cfg.TRAIN.SELF_SUPERVISED)
        

        train, val, test = random_split(full_dataset, (0.7, 0.15, 0.15))
        dataset = {'train': train, 'val': val, 'test': test}
        
        self.dataloader = {x: get_dataloader(dataset[x],
                                            shuffle=True,
                                            fix_seed=(x != 'train'),
                                            batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                            seed=self.seed,
                                            num_workers=4)
                        for x in ('train', 'val', 'test')}

        # Train and eval model
        if self.cfg.TRAIN.TRAIN:
            self.train_eval_model()
        
        # Test model
        if self.cfg.EVAL.TEST:
            test_acc = self.eval_model(mode='test')

        return self.S


    def train_eval_model(self):
        """
        Train the model.
        """
        
        # ------------------------ 
        #  Training configuration
        # ------------------------
        
        # Loss function
        if self.cfg.TRAIN.LOSS_FUNC.lower() == 'cml':
            self.criterion = ContrastiveLossWithAttention()
        elif self.cfg.TRAIN.LOSS_FUNC.lower() == 'distill_qc':
            print("[SHELLEY] You selected 'distill_quadratic' as loss function which is defined within the model,"
                  "please ensure there is a tensor with key 'loss' in your model\'s returned dictionary.")
            self.criterion = None
        else:
            raise Exception(f"[SHELLEY] Invalid loss function: {self.cfg.TRAIN.LOSS_FUNC.lower()}.")
        
        # Get model parameters
        if self.cfg.TRAIN.SEPARATE_BACKBONE_LR:
            backbone_ids = [id(item) for item in self.model.backbone_params]
            other_params = [param for param in self.model.parameters() if id(param) not in backbone_ids]

            model_params = [
                {'params': other_params},
                {'params': self.model.backbone_params, 'lr': self.cfg.TRAIN.BACKBONE_LR}
            ]

        else:
            model_params = self.model.parameters()
        
        # Optimizer
        if self.cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
            self.optimizer = optim.SGD(model_params, lr=self.cfg.TRAIN.LR, momentum=self.cfg.TRAIN.MOMENTUM, nesterov=True)
        elif self.cfg.TRAIN.OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(model_params, lr=self.cfg.TRAIN.LR)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.TRAIN.OPTIMIZER}.")
        
        # Scheduler
        if self.cfg.TRAIN.USE_SCHEDULER:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.cfg.TRAIN.LR_STEP,
                gamma=self.cfg.TRAIN.LR_DECAY,
                last_epoch=self.cfg.TRAIN.START_EPOCH - 1
            )

        # Number of iterations per epoch
        if self.cfg.TRAIN.EPOCH_ITERS < 0:
            # Iterate all over the training dataset
            self.cfg.TRAIN.EPOCH_ITERS = len(self.dataloader['train'])
            print('[SHELLEY] epoch_iters: ', self.cfg.TRAIN.EPOCH_ITERS)

            if self.cfg.HEAD.NAME == 'common':
                if self.cfg.HEAD.WARMUP_STEP < 0:
                    # Set the warmup steps equal to the epoch iterations
                    self.cfg.HEAD.WARMUP_STEP = self.cfg.TRAIN.EPOCH_ITERS
                    print('[SHELLEY] warmup_step: ', self.cfg.TRAIN.EPOCH_ITERS)
        
        # -------------------
        #  Train / eval loop
        # -------------------
        print("Start training...")
        since = time.time()
        start_epoch = self.cfg.TRAIN.START_EPOCH
        num_epochs = self.cfg.TRAIN.NUM_EPOCHS

        # Early stopping parameters
        best_val_acc = 0.0
        best_model_path = f'{self.cfg.CHECKPOINTS}/{self.cfg.BACKBONE.NAME}_{self.cfg.HEAD.NAME}_{self.cfg.DATA.NAME}_add{self.cfg.DATA.P_ADD}_rm{self.cfg.DATA.P_RM}.pth'
        no_improvement_count = 0

        for epoch in range(start_epoch, num_epochs):
            # Reset seed after evaluation per epoch
            torch.manual_seed(self.seed + epoch + 1)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # Train model
            train_loss, train_acc = self.train_one_epoch(epoch)
            print(f"[TRAINING] Epoch {epoch+1:<4} Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}")

            # Validation    
            if self.cfg.EVAL.VALIDATE:  
                val_acc = self.eval_model(mode='val')

                # Check for improvement
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improvement_count = 0

                    # Save best model
                    torch.save(self.model.state_dict(), best_model_path)

                    print(f"New best model saved with validation accuracy: {val_acc:.4f}")
                else:
                    no_improvement_count += 1

                # Early stopping
                if self.cfg.TRAIN.EARLY_STOPPING and no_improvement_count >= self.cfg.TRAIN.PATIENCE:
                    print(f"Early stopping after {epoch+1} epochs with no improvement in validation loss.")
                    break

            # Update scheduler
            if self.cfg.TRAIN.USE_SCHEDULER:
                self.scheduler.step()
        
        # Compute elapsed training time
        time_elapsed = time.time() - since 
        hours = time_elapsed // 3600
        minutes = (time_elapsed // 60) % 60
        seconds = time_elapsed % 60
        print(f"Training complete in {hours:.0f}h {minutes:0f}m {seconds:0f}s")

        # Load best model parameters
        self.model.load_state_dict(torch.load(best_model_path))
        self.best_model_path = best_model_path

        return
    

    def train_one_epoch(self, epoch):
        # Set model to training mode
        self.model.train()

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in self.optimizer.param_groups]))

        epoch_loss = 0.0
        epoch_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data
        for inputs in self.dataloader['train']:

            if iter_num >= self.cfg.TRAIN.EPOCH_ITERS:
                break
            
            # Move data to correct device
            if self.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)
            
            iter_num += 1

            # Zero the gradient parameters
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward step
                if 'common' in self.cfg.HEAD.NAME: # COMMON use the iter number to control the warmup temperature
                    outputs = self.model(inputs, training=True, iter_num=iter_num, epoch=epoch)
                else:
                    outputs = self.model(inputs)

                # Compute loss
                if self.cfg.TRAIN.LOSS_FUNC == 'cml':
                    loss = self.criterion(outputs['ds_mat'], outputs['gt_perm_mat'], outputs['perm_mat'], *outputs['ns'], beta=0.1)
                elif self.cfg.TRAIN.LOSS_FUNC == 'distill_qc':
                    loss = torch.sum(outputs['loss'])
                else:
                    raise ValueError(f"[SHELLEY] Unsupperted loss function: {self.cfg.TRAIN.LOSS_FUNC}")

                # Compute accuracy
                acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)[0]

                # Backward step
                loss.backward()
                self.optimizer.step()

                batch_num = inputs['batch_size']

                # Batching statistics
                running_loss += loss.item() * batch_num
                running_acc += acc.item() * batch_num
                epoch_loss += loss.item() * batch_num
                epoch_acc += acc.item() * batch_num
                

                if iter_num % self.cfg.TRAIN.STATISTIC_STEP == 0:
                    running_speed = self.cfg.TRAIN.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print(f'Epoch: {epoch+1:<4}' \
                            f'Iteration: {iter_num:<4}' \
                            f'{running_speed:<4.2f}sample/s'\
                            f'Loss: {running_loss / self.cfg.TRAIN.STATISTIC_STEP / batch_num:<8.4f}'\
                            f'Accuracy: {running_acc / self.cfg.TRAIN.STATISTIC_STEP / batch_num:<8.4f}')

                    running_loss = 0.0
                    running_acc = 0.0
                    running_since = time.time()
    
        # Epoch statistics
        epoch_loss = epoch_loss / iter_num / batch_num
        epoch_acc = epoch_acc / iter_num / batch_num

        return epoch_loss, epoch_acc
    

    def eval_model(self, mode='val'):
        """
        Evaluate the model on validation or test set and return the corresponding metrics.
        """
        self.model.eval()        
        dataloader = self.dataloader[mode]  # 'val' or 'test'

        eval_acc = 0.0
        eval_since = time.time()
        eval_iter_num = 0

        # Iterate over data
        with torch.no_grad():
            for eval_inputs in dataloader:
                if eval_iter_num >= self.cfg.EVAL.VAL_ITERS and mode == 'val':
                    break
                elif eval_iter_num >= self.cfg.EVAL.TEST_ITERS and mode == 'test':
                    break

                # Move data to correct device
                if self.model.device != torch.device('cpu'):
                    eval_inputs = data_to_cuda(eval_inputs)

                eval_iter_num += 1

                # Forward step
                eval_outputs = self.model(eval_inputs, training=False)

                # Compute accuracy
                eval_acc += matching_accuracy(eval_outputs['perm_mat'], eval_outputs['gt_perm_mat'], eval_outputs['ns'], idx=0)[0].item()

        # Average accuracy
        eval_acc = eval_acc / eval_iter_num

        # Print evaluation statistics
        print(f"[{mode.upper()}] Accuracy: {eval_acc:.4f}")

        # Print time taken for evaluation
        eval_time_elapsed = time.time() - eval_since
        print(f"Evaluation complete in {eval_time_elapsed:.0f}s")

        return eval_acc