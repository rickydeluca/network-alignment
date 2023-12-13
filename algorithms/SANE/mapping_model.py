import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =======
#   DNN
# =======
class DNNModule(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super(DNNModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)
    
class LinkPredictor(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super(LinkPredictor, self).__init__()
        self.dnn1 = DNNModule(input_dim, dropout=dropout)
        self.dnn2 = DNNModule(input_dim, dropout=dropout)

        self.joint_module = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def pred(self, x_i, x_j):
        x_i = self.dnn1(x_i)
        x_j = self.dnn2(x_j)
        x_out = torch.cat([x_i, x_j], dim=1)
        x_out = self.joint_module(x_out)
        return x_out.squeeze(1)

    def forward(self, x_i, x_j):
        x_out = self.pred(x_i, x_j)
        return F.softmax(x_out, dim=0)
    

# =====================
#   COSINE SIMILARITY
# =====================
class CosineSimilarity(nn.Module):
    """
    Wrapper class for the pytorch `CosineSimilarity` class.
    """
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    def pred(self, x_i, x_j):
        return self.cosine_similarity(x_i, x_j)
    
    
# =================
#   INNER PRODUCT
# =================
class InnerProduct(nn.Module):
    """
    Wrapper class to compute the inner product.
    """
    def __init__(self):
        super(InnerProduct, self).__init__()
        
    def pred(self, x_i, x_j):
        x_i_norm = F.normalize(x_i, p=2, dim=-1)
        x_j_norm = F.normalize(x_j, p=2, dim=-1)
        return (x_i_norm * x_j_norm).sum(dim=-1)

# ===================
#   CROSS ATTENTION
# ===================

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


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super(TransformerBlock, self).__init__()

        self.cross_attention = CrossAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x1, x2):
        attended_x1_x2, attended_x2_x1 = self.cross_attention(x1, x2)
        x1 = self.norm1(attended_x1_x2 + x1)
        x2 = self.norm1(attended_x2_x1 + x2)

        fedforward_x1 = self.ff(x1)
        fedforward_x2 = self.ff(x2)

        return self.norm2(fedforward_x1 + x1), self.norm2(fedforward_x2 + x2)
