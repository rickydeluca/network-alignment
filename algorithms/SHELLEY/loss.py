import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Distill_InfoNCE(nn.Module):
    def __init__(self):
        super(Distill_InfoNCE, self).__init__()

    def forward(self, feature: Tensor, feature_m: Tensor, alpha: float, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor, groundtruth:Tensor) -> Tensor:

        graph1_feat = F.normalize(feature[0], dim=-1)
        graph2_feat = F.normalize(feature[1], dim=-1)

        # following the contrastive in "Learning Transferable Visual Models From Natural Language Supervision"
        sim_1to2 = dynamic_temperature.exp() * graph1_feat @ graph2_feat.T
        sim_2to1 = dynamic_temperature.exp() * graph2_feat @ graph1_feat.T

        # get momentum features
        with torch.no_grad():
            graph1_feat_m = F.normalize(feature_m[0], dim=-1)
            graph2_feat_m = F.normalize(feature_m[1], dim=-1)

            # momentum similiarity
            sim_1to2_m = dynamic_temperature_m.exp() * graph1_feat_m @ graph2_feat_m.T
            sim_2to1_m = dynamic_temperature_m.exp() * graph2_feat_m @ graph1_feat_m.T
            sim_1to2_m = F.softmax(sim_1to2_m, dim=1)
            sim_2to1_m = F.softmax(sim_2to1_m, dim=1)

            # online similiarity
            # sim_targets = torch.zeros(sim_1to2_m.size()).to(graph1_feat.device)
            # sim_targets.fill_diagonal_(1)
            sim_targets = groundtruth

            # generate pseudo contrastive labels
            sim_1to2_targets = alpha * sim_1to2_m + (1 - alpha) * sim_targets
            sim_2to1_targets = alpha * sim_2to1_m + (1 - alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_1to2, dim=1) * sim_1to2_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_2to1, dim=1) * sim_2to1_targets, dim=1).mean()
        contrast_loss = (loss_i2t + loss_t2i) / 2
        return contrast_loss


class Distill_QuadraticContrast(nn.Module):
    def __init__(self):
        super(Distill_QuadraticContrast, self).__init__()

    def normalize(self, x: Tensor):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def forward(self, feature: Tensor, feature_m: Tensor, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor) -> Tensor:
        graph1_feat = F.normalize(feature[0], dim=-1)
        graph2_feat = F.normalize(feature[1], dim=-1)
        batch_size = graph1_feat.shape[0]

        with torch.no_grad():
            graph1_feat_m = F.normalize(feature_m[0], dim=-1)
            graph2_feat_m = F.normalize(feature_m[1], dim=-1)
            sim_1to2_m = graph1_feat_m @ graph2_feat_m.T
            w = ((torch.diag(sim_1to2_m) / sim_1to2_m.sum(dim=1)) + (
                        torch.diag(sim_1to2_m) / sim_1to2_m.sum(dim=0))) / 2
            # normalize w
            w = self.normalize(w)
            w = torch.mm(w.unsqueeze(1), w.unsqueeze(0))
            w = self.normalize(w)

        # cross-graph similarity
        sim_1to2 = dynamic_temperature.exp() * graph1_feat @ graph2_feat.T
        sim_2to1 = dynamic_temperature.exp() * graph2_feat @ graph1_feat.T
        # within-graph similarity
        sim_1to1 = dynamic_temperature.exp() * graph1_feat @ graph1_feat.T
        sim_2to2 = dynamic_temperature.exp() * graph2_feat @ graph2_feat.T
        # within-graph consistency
        within_graph_loss = (w * (sim_1to1 - sim_2to2).square()).mean() * batch_size / \
                            (dynamic_temperature.exp() * dynamic_temperature.exp()) # using batch_size to scale the loss
        # cross-graph consistency
        cross_graph_loss = (w * (sim_1to2 - sim_2to1).square()).mean() * batch_size / \
                           (dynamic_temperature.exp() * dynamic_temperature.exp())
        graph_loss = within_graph_loss + cross_graph_loss

        return graph_loss
    

class CommonLoss(object):
    def __init__(self):
        self.contrast_loss = Distill_InfoNCE()
        self.graph_loss = Distill_QuadraticContrast()

    def forward(self, args):
        
        feature = args.feature
        feature_m = args.feature_m
        alpha = args.alpha
        dynamic_temperature = args.dynamic_temperature
        dynamic_temperature_m = args.dynamic_temperature_m
        groundtruth = args.groundtruth
        
        return  self.contrast_loss(feature, feature_m, alpha, dynamic_temperature, dynamic_temperature_m, groundtruth) + \
                self.graph_loss(feature, feature_m, dynamic_temperature, dynamic_temperature_m)


def get_loss_function(name: str):
    """
    Return the loss function defined by the string `name`.
    """

    if name == 'common':
        return CommonLoss()