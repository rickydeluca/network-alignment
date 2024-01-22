import torch
import torch.nn.functional as F
from torch import Tensor


# =============
#   EMBEDDING
# =============

class EmbeddingLossFunctions(object):
    def __init__(self, loss_fn='xent', neg_sample_weights=1.0):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be
                based on dot product.
        """
        self.neg_sample_weights = neg_sample_weights
        self.output_dim = 1
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        else:
            print("Not implemented yet.")


    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [n_batch_edges x feature_size].
        """
        # shape: [n_batch_edges, input_dim1]
        result = torch.sum(inputs1 * inputs2, dim=1) # shape: (n_batch_edges,)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [n_batch_edges x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        neg_aff = inputs1.mm(neg_samples.t()) #(n_batch_edges, num_neg_samples)
        return neg_aff


    def sigmoid_cross_entropy_with_logits(self, labels, logits):
        sig_aff = torch.sigmoid(logits)
        loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)
        return loss

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        inputs1: Tensor (512, 256), normalized vector
        inputs2: Tensor (512, 256), normalized vector
        neg_sample: Tensor (20, 256)
        """
        cuda = inputs1.is_cuda
        true_aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples)
        true_labels = torch.ones(true_aff.shape)  # (n_batch_edges,)
        if cuda:
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if cuda:
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        loss0 = true_xent.sum()
        loss1 = self.neg_sample_weights * neg_xent.sum()
        loss = loss0 + loss1
        return loss, loss0, loss1



# ===========
#   MAPPING
# ===========
    
class Distill_InfoNCE(torch.nn.Module):
    def __init__(self):
        super(Distill_InfoNCE, self).__init__()

    def forward(self, feature: Tensor, feature_m: Tensor, alpha: float, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor, groundtruth:Tensor) -> Tensor:
        
        # DEBUG: Verify devices
        # print("Loss device:")
        # print('source feature: ', feature[0].device)
        # print('target feature: ', feature[1].device)
        # print('source feature_m: ', feature_m[0].device)
        # print('target feature_m: ', feature_m[1].device)
        # print('dynamic_temperature: ', dynamic_temperature.device)
        # print('dynamic_temperature_m: ', dynamic_temperature_m.device)
        # print('groundtruth: ', groundtruth.device)

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


class Distill_QuadraticContrast(torch.nn.Module):
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


class MappingLossFunctions(object):
    def __init__(self):
        self.contrast_loss = Distill_InfoNCE()
        self.graph_loss = Distill_QuadraticContrast()

    def loss(self, feature=None, feature_m=None, alpha=None, 
             dynamic_temperature=None, dynamic_temperature_m=None, 
             groundtruth=None):
        
        loss =  self.contrast_loss(feature, feature_m, alpha, dynamic_temperature, dynamic_temperature_m, groundtruth) + \
                self.graph_loss(feature, feature_m, dynamic_temperature, dynamic_temperature_m)
        
        return loss
        