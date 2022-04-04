import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Variational Graph Auto-Encoder (VGAE)
# Origitnal paper:
# Kipf, T. N. and Welling, M. 'Variational graph autoencoders'

class VGAE(nn.Module):  # edge probability calculation using GCN
    def __init__(self, feature_dim, h_dim, z_dim, activation, A_adj):
        # feaature_dim: dimension of feature matrix
        # h_dim: dimension of H
        # z_dim: dimension of Z
        # activation: function (e.g., sigmoid or relu)
        # A_adj: adjacent matrix of the graph
        super(VGAE, self).__init__()

        # make GCN Layer 
        self.gcn_base = GCNLayer(feature_dim, h_dim, None)
        self.gcn_mean = GCNLayer(h_dim, z_dim, activation)
        self.gcn_log_std = GCNLayer(h_dim, z_dim, activation)

    def forward(self, A_adj, features):
        # A_adj: adjacent matrix of the graph
        # features: feature matrix

        # GCN encoder
        hidden_layer = self.gcn_base(A_adj, features)
        self.mean = self.gcn_mean(A_adj, hidden_layer)
        self.log_std = self.gcn_log_std(A_adj, hidden_layer)
        # add Gaussian random noise
        noise = torch.randn_like(self.mean)
        Z = noise*torch.exp(self.log_std) + self.mean

        # decode (inner product) and return edge probablity
        A_pred = Z @ Z.T
        return A_pred

# Make GCN Layer for NN
class GCNLayer(nn.Module):  # single layer of GCN
    def __init__(self, in_dim, out_dim, activation):
        super(GCNLayer, self).__init__()
        # Set weight matrix (W)
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.avtivation = activation
        # for param in self.parameters():
            # nn.init.xavier_uniform_(param)

    def forward(self, adj, h):
        # sigmoid(AHW) (A: normalized A)
        X = h @ self.W
        X = adj @ X
        if self.activation:
            x = self.activation(X)
        return X

