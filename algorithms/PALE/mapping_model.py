import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.PALE.loss import MappingLossFunctions

##############################################################################
#               MAPPING MODELS
##############################################################################

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


class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, graph_A, graph_B):
        # Assuming graph_A and graph_B are node embeddings (N x D)
        # Transpose to make it suitable for attention mechanism
        graph_A = graph_A.unsqueeze(0)
        graph_B = graph_B.unsqueeze(0)

        # Apply cross-attention
        output_A, _ = self.attention(graph_A, graph_B, graph_B)
        output_B, _ = self.attention(graph_B, graph_A, graph_A)
        
        # Transpose back to original shape
        output_A = output_A.squeeze(0)
        output_B = output_B.squeeze(0)
        
        return output_A, output_B


class PaleMappingCrossTransformer(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding, num_attention_layers=1, num_ff_layers=1, heads=1, activate_function='sigmoid'):
        super(PaleMappingCrossTransformer, self).__init__(source_embedding, target_embedding)

        # Activation function
        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()

        # Cross-Attention layers
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionLayer(embedding_dim, num_heads=heads) for _ in range(num_attention_layers)]
        )

        # Feedforward layer
        if num_ff_layers == 1:
            self.ff = nn.Linear(embedding_dim, embedding_dim, bias=True)
        elif num_ff_layers == 2:
            hidden_dim = 2*embedding_dim
            self.ff = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim, bias=True),
                self.activate_function,
                nn.Linear(hidden_dim, embedding_dim, bias=True)
            )
        else:
            raise ValueError("Max 2 ff layers.")
        
    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping, target_feats_after_mapping = self.forward(source_feats, target_feats)
        
        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats_after_mapping) / batch_size

        return mapping_loss
    
    def forward(self, x1, x2):
        # Cross-Attention layers
        for cross_attention_layer in self.cross_attention_layers:
            x1, x2 = cross_attention_layer(x1, x2)
        
        # Feed-forward layers
        x1 = self.ff(x1)
        x2 = self.ff(x2)

        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        return x1, x2

