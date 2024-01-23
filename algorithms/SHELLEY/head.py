import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.SHELLEY.backbone import get_backbone
from algorithms.SHELLEY.loss import get_loss_function

class InnerProduct(nn.Module):
    def __init__(self, output_dim):
        super(InnerProduct, self).__init__()
        self.d = output_dim

    def forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        X = torch.nn.functional.normalize(X, dim=-1)
        Y = torch.nn.functional.normalize(Y, dim=-1)
        res = torch.matmul(X, Y.transpose(0, 1))
        return res


class Net(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg:    Dictionary of dictionaries (easydict) that contains
                    the configuration for the main model and the backbone.

        Returns:
            model:  The trained model.
        """
        super(Net, self).__init__()

        # model parameters
        self.online_net = get_backbone(name=cfg.BACKBONE.NAME, args=cfg.BACKBONE)
        self.momentum_net = get_backbone(name=cfg.BACKBONE.NAME, args=cfg.BACKBONE)
        self.momentum = cfg.MODEL.MOMENTUM          # for momentum network
        self.warmup_step = cfg.MODEL.WARMUP_STEP    # to reach the final alpha
        self.epoch_iters = cfg.MODEL.EPOCH_ITERS    
        self.final_alpha = cfg.MODEL.ALPHA          # target after warmup step
        self.distill = cfg.MODEL.DISTILL            # boolean

        self.model_pairs = [self.online_net, self.momentum_net]
        self.copy_params()  # init momentum network

        self.vertex_affinity = InnerProduct(cfg.BACKBONE.OUT_CHANNELS)

        # define loss function
        self.loss_fn_name = cfg.TRAIN.LOSS
        self.loss_fn = get_loss_function(name=self.loss_fn_name)

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # compute the distillation weight `alpha`
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = self.final_alpha
        else:
            alpha = self.final_alpha * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)
        
        # get output of the online network
        embedding_list = self.online_net(data_dict) # [source_emb_tensor, target_emb_tensor]

        # generate node affinity matrix using the node embeddings
        aff_mat = self.vertex_affinity(embedding_list[0], embedding_list[1])

        # generate the permutation matrix (the predicted alignments)
        perm_mat = None     # TODO

        if training is True:
            # the momentum network is used only during training
            assert self.distill

            # get output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                embedding_m_list = self.momentum_net(data_dict)

            # compute loss
            if self.loss_fn_name == 'common':
                loss_args = {'feature': embedding_list,
                            'feature_m': embedding_m_list,
                            'dynamic_temperature': self.online_net.logit_scale,
                            'dynamic_temperature_m': self.momentum_net.logit_scale,
                            'groundtruth': data_dict.groundtruth}
            else:
                raise Exception(f"Invalid loss: {self.loss_fn_name}")
                
            loss = self.loss_fn(loss_args)

            # update the dictionary
            data_dict.update({
                'perm_mat': perm_mat,
                'loss': loss,
                'ds_mat': None
            })

        else:  # directly output the predicted alignment
            data_dict.update({
                'perm_mat': perm_mat,
                'ds_mat': None
            })

        return data_dict

        
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False   # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)