import torch
import torch.nn as nn

from algorithms.SHELLEY.backbone import get_backbone
from algorithms.SHELLEY.loss import get_loss_function
from algorithms.SHELLEY.utils.lap_solvers import hungarian


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


class CommonHead(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg:    Dictionary of dictionaries (easydict) that contains
                    the configuration for the main model and the backbone.

        Returns:
            model:  The trained model.
        """
        super(CommonHead, self).__init__()

        # model parameters
        self.online_net = get_backbone(name=cfg.backbone.name, cfg=cfg.backbone)
        self.momentum_net = get_backbone(name=cfg.backbone.name, cfg=cfg.backbone)
        self.momentum = cfg.model.momentum          # for momentum network
        self.warmup_step = cfg.model.warmup_step    # to reach the final alpha
        self.epoch_iters = cfg.model.epoch_iters    
        self.final_alpha = cfg.model.alpha          # target after warmup step
        self.distill = cfg.model.distill            # boolean

        self.model_pairs = [[self.online_net, self.momentum_net]]
        self.copy_params()  # init momentum network

        self.vertex_affinity = InnerProduct(1536)   # DEBUG

        # define loss function
        self.loss_fn = get_loss_function(name='common')

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        # compute the distillation weight `alpha`
        if epoch * self.epoch_iters + iter_num >= self.warmup_step:
            alpha = self.final_alpha
        else:
            alpha = self.final_alpha * min(1, (epoch * self.epoch_iters + iter_num) / self.warmup_step)
        
        # get output of the online network
        embedding_list = self.online_net(data_dict) # out: [source_emb_tensor, target_emb_tensor]

        # generate node affinity matrix using the node embeddings
        aff_mat = self.vertex_affinity(embedding_list[0], embedding_list[1])

        # generate the permutation matrix (the predicted alignments)
        num_source_nodes = torch.tensor(aff_mat.shape[0]).unsqueeze(0)
        num_target_nodes = torch.tensor(aff_mat.shape[1]).unsqueeze(0)
        perm_mat = hungarian(aff_mat, num_source_nodes, num_target_nodes)

        if training is True:
            # the momentum network is used only during training
            assert self.distill

            # get output of the momentum network
            with torch.no_grad():
                self._momentum_update()
                embedding_m_list = self.momentum_net(data_dict)
                
            loss = self.loss_fn(embedding_list,
                                embedding_m_list,
                                alpha,
                                self.online_net.logit_scale,
                                self.momentum_net.logit_scale,
                                data_dict.groundtruth)

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



def get_head(name, cfg):
    if name == 'common':
        model = CommonHead(cfg)
    else:
        raise Exception(f"Invalid head: {name}.")
    
    return model