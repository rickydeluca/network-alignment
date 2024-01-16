import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.SANE.loss import MappingLossFunctions

class PaleMapping(nn.Module):
    def __init__(self, source_embedding, target_embedding):
        """
        Parameters
        ----------
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target_nodes
        target_neighbor: dict
            dict of target_node -> target_nodes_neighbors. Used for calculate vinh_loss
        """

        super(PaleMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.loss_fn = MappingLossFunctions()
    

class PaleMappingLinear(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding):
        super(PaleMappingLinear, self).__init__(source_embedding, target_embedding)
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        
        return mapping_loss

    def forward(self, source_feats):
        ret = self.maps(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret


class PaleMappingMlp(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding, activate_function='sigmoid'):

        super(PaleMappingMlp, self).__init__(source_embedding, target_embedding)

        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()

        hidden_dim = 2*embedding_dim
        self.mlp = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            self.activate_function,
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])


    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        

        return mapping_loss

    def forward(self, source_feats):
        ret = self.mlp(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret


class CrossAttention(nn.Module):
    def __init__(self, m, heads=8):
        super(CrossAttention, self).__init__()

        self.heads = heads
        
        # Linear transformations for keys, queries, and values for both inputs
        self.tokeys_x1 = nn.Linear(m, m * heads, bias=False)
        self.toqueries_x1 = nn.Linear(m, m * heads, bias=False)
        self.tovalues_x1 = nn.Linear(m, m * heads, bias=False)

        self.tokeys_x2 = nn.Linear(m, m * heads, bias=False)
        self.toqueries_x2 = nn.Linear(m, m * heads, bias=False)
        self.tovalues_x2 = nn.Linear(m, m * heads, bias=False)

        # The final linear transformation to finish with m-dimensional vectors
        self.mergeheads_x1 = nn.Linear(heads * m, m)
        self.mergeheads_x2 = nn.Linear(heads * m, m)

    def forward(self, x1, x2):
        b1, t1, m = x1.size()  # Batch dimension, sequence length, input vector dimension for x1
        b2, t2, _ = x2.size()  # Batch dimension, sequence length, input vector dimension for x2
        r = self.heads

        # Transformations for x1
        keys_x1 = self.tokeys_x1(x1).view(b1, t1, r, m)
        queries_x1 = self.toqueries_x1(x1).view(b1, t1, r, m)
        values_x1 = self.tovalues_x1(x1).view(b1, t1, r, m)

        # Transformations for x2
        keys_x2 = self.tokeys_x2(x2).view(b2, t2, r, m)
        queries_x2 = self.toqueries_x2(x2).view(b2, t2, r, m)
        values_x2 = self.tovalues_x2(x2).view(b2, t2, r, m)

        # Cross-attention from x1 to x2
        w_prime_x1_x2 = torch.einsum('btrm,bfrm->brtf', queries_x1, keys_x2) / math.sqrt(m)
        w_x1_x2 = F.softmax(w_prime_x1_x2, dim=-1)
        y_conc_x1_x2 = torch.einsum('brtf,bfrm->btrm', w_x1_x2, values_x2)
        y_x1_x2 = torch.einsum('btrm,krm->btk', y_conc_x1_x2, self.mergeheads_x2.weight.view(m, r, m)) + self.mergeheads_x2.bias

        # Cross-attention from x2 to x1
        w_prime_x2_x1 = torch.einsum('btrm,bfrm->brtf', queries_x2, keys_x1) / math.sqrt(m)
        w_x2_x1 = F.softmax(w_prime_x2_x1, dim=-1)
        y_conc_x2_x1 = torch.einsum('brtf,bfrm->btrm', w_x2_x1, values_x1)
        y_x2_x1 = torch.einsum('btrm,krm->btk', y_conc_x2_x1, self.mergeheads_x1.weight.view(m, r, m)) + self.mergeheads_x1.bias

        return y_x1_x2, y_x2_x1


class PaleMappingCrossTransformer(PaleMapping):
    def __init__(self, source_embedding, target_embedding, k=None, heads=1, activate_function='sigmoid',):
        super(PaleMappingCrossTransformer, self).__init__(source_embedding, target_embedding)

        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()

        self.cross_attention = CrossAttention(k, heads=heads)

        self.norm1 = nn.BatchNorm1d(k)
        self.norm2 = nn.BatchNorm1d(k)

        hidden_dim = 4*k
        self.ff = nn.Sequential(
            nn.Linear(k, hidden_dim, bias=True),
            self.activate_function,
            nn.Linear(hidden_dim, k, bias=True)
        )

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping, target_feats_after_mapping = self.forward(source_feats, target_feats)
        
        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats_after_mapping) / batch_size

        return mapping_loss
    
    def forward(self, x1, x2):
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        attended_x1_x2, attended_x2_x1 = self.cross_attention(x1, x2)
        x1 = self.norm1(attended_x1_x2 + x1).squeeze(0)
        x2 = self.norm1(attended_x2_x1 + x2).squeeze(0)

        fedforward_x1 = self.ff(x1)
        fedforward_x2 = self.ff(x2)

        x1 = self.norm2(fedforward_x1 + x1)
        x2 = self.norm2(fedforward_x2 + x2)

        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        return x1, x2

