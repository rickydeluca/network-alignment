import torch
import torch.nn as nn
import torch.nn.functional as F


# =======
#   DNN
# =======
class DNNModule(nn.Module):
    def __init__(self, input_dim):
        super(DNNModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.layers(x)
    
class LinkPredictor(nn.Module):
    def __init__(self, input_dim):
        super(LinkPredictor, self).__init__()
        self.dnn1 = DNNModule(input_dim)
        self.dnn2 = DNNModule(input_dim)

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
        return F.softmax(x_out, dim=1)
    
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
        return (x_i * x_j).sum(dim=-1)