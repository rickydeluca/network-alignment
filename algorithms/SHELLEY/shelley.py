import numpy as np
import torch
import torch.optim as optim
from easydict import EasyDict as edict

import evaluation.metrics as metrics
from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.SHELLEY.head import get_head
from algorithms.SHELLEY.utils.pyg_convertion import networkx_to_pyg
from algorithms.SHELLEY.utils.split_dict import shuffle_and_split
from utils.graph_utils import load_gt


class SHELLEY(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.train_eval_dict = args.train_dict
        self.use_pretrained = args.use_pretrained
        self.skip_train = args.skip_train
        self.seed = args.seed
        self.gt_mat = torch.from_numpy(load_gt(self.train_eval_dict, source_dataset.id2idx, target_dataset.id2idx, 'matrix'))
        self.S = None

        if args.eval:
            # split the training dictionary in half and use the
            # first half for training and second for evalutation
            gt_dict = load_gt(self.train_eval_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
            train_dict, eval_dict = shuffle_and_split(gt_dict)
            
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

        # head
        self.cfg.model = edict()
        self.cfg.model.name = args.head

        if args.head == 'sigma':
            pass # TODO
        elif args.head == 'common':
            self.cfg.model.distill = args.distill
            self.cfg.model.momentum = args.distill_momentum
            # retrieve warmup_step from the `batchsize` argument if `args.warmup_step` == -1
            self.cfg.model.warmup_step = len(self.source_train_nodes) // args.batchsize if args.warmup_step == -1 else args.warmup_step 
            self.cfg.model.epoch_iters = args.epoch_iters
            self.cfg.model.alpha = args.alpha
        else:
            raise Exception(f"Invalid head: {self.head}.")
        
        # training configuration
        self.cfg.train = edict()
        self.cfg.train.optimizer = args.optimizer
        self.cfg.train.momentum = args.optim_momentum 
        self.cfg.train.lr = args.lr
        self.cfg.train.use_scheduler = args.use_scheduler
        self.cfg.train.lr_step = args.lr_step 
        self.cfg.train.lr_decay = args.lr_decay
        self.cfg.train.start_epoch = args.start_epoch
        self.cfg.train.num_epochs = args.num_epochs
        # do not perform mini-batching if `args.batchsize` == -1
        self.cfg.train.batchsize = len(self.source_train_nodes) if args.batchsize == -1 else args.batchsize


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
        if not self.skip_train:
            self.train_eval_model()
        
        # get final alignment
        self.model.eval()

        inputs = edict()
        inputs.source_graph = self.source_graph
        inputs.target_graph = self.target_graph

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

        # optimizer
        if self.cfg.train.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model_params, lr=self.cfg.train.lr, momentum=self.cfg.train.momentum, nesterov=True)
        elif self.cfg.train.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model_params, lr=self.cfg.train.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.train.optimizer}.")
        
        # scheduler
        if self.cfg.train.use_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.train.lr_step,
                gamma=self.cfg.train.lr_decay,
                last_epoch=self.cfg.train.start_epoch - 1
            )
        
        # ---------------
        # train/eval loop
        # ---------------
        start_epoch = self.cfg.train.start_epoch
        num_epochs = self.cfg.train.num_epochs
        batch_size = self.cfg.train.batchsize
        mb_iters = len(self.source_train_nodes) // batch_size
        assert mb_iters > 0, "batch_size is too large"
        if(len(self.source_train_nodes) % batch_size > 0):
            mb_iters += 1

        for epoch in range(start_epoch, num_epochs):
            # reset seed
            torch.manual_seed(self.seed + epoch + 1)

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
                # get batch nodes
                source_batch = self.source_train_nodes[iter*batch_size:(iter+1)*batch_size]
                target_batch = [self.gt_train_dict[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch).to(self.device)
                target_batch = torch.LongTensor(target_batch).to(self.device)

                # define inputs for the model
                inputs = edict()
                # inputs.source_graph = self.source_graph.subgraph(source_batch).to(self.device)  # NOTE: creating subgraphs might not be the answer
                # inputs.target_graph = self.target_graph.subgraph(target_batch).to(self.device)
                inputs.source_graph = self.source_graph.to(self.device)
                inputs.target_graph = self.target_graph.to(self.device)
                inputs.source_batch = source_batch
                inputs.target_batch = target_batch
                inputs.gt_perm_mat = self.gt_mat.to(self.device)[source_batch, target_batch]    # NOTE: is it correct?

                # zero gradient params
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward step
                    if 'common' in self.cfg.model.name:
                        outputs = self.model(inputs, training=True, iter_num=iter, epoch=epoch)
                    else:
                        raise Exception(f"Invalid model head {self.cfg.model.name}")
                    
                    # check output
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs

                    # loss
                    # loss = torch.sum(outputs.loss)  # TODO
                    loss = outputs.loss

                    # backward step
                    loss.backward()
                    optimizer.step()

                    print(f"train loss: {loss}")

            # evalutation:
            self.model.eval()   # set model to eval mode
            # np.random.shuffle(self.source_eval_nodes)   # NOTE: maybe not?
            with torch.no_grad():
                source_batch = self.source_eval_nodes
                target_batch = [self.gt_eval_dict[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch).to(self.device)
                target_batch = torch.LongTensor(target_batch).to(self.device)

                inputs = edict()
                # inputs.source_graph = self.source_graph.subgraph(source_batch).to(self.device)
                # inputs.target_graph = self.target_graph.subgraph(target_batch).to(self.device)
                inputs.source_graph = self.source_graph.to(self.device)
                inputs.target_graph = self.target_graph.to(self.device)
                inputs.source_batch = source_batch
                inputs.target_batch = target_batch
                inputs.gt_perm_mat = self.gt_mat.to(self.device)[source_batch, target_batch]
                
                if 'common' in self.cfg.model.name:
                    outputs = self.model(inputs, training=False)
                else:
                    raise Exception(f"Invalid model head {self.cfg.model.name}")
                
                # compute matching accuracy on eval subset
                pred = metrics.greedy_match(outputs.perm_mat.detach().cpu().numpy())
                # pred = outputs.perm_mat.cpu().numpy()
                match_acc = metrics.compute_accuracy(pred, self.gt_mat)

                print(f'eval matching accuracy: {match_acc:.4f}')

            # update scheduler
            if self.cfg.train.use_scheduler:
                scheduler.step()