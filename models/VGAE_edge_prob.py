import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

# Variational Graph Auto-Encoder (VGAE)
# Origitnal paper:
# Kipf, T. N. and Welling, M. 'Variational graph autoencoders'

class VGAE(nn.Module):  # edge probability calculation using GCN
    def __init__(self, feature_dim, h_dim, z_dim, activation):
        # feaature_dim: dimension of feature matrix
        # h_dim: dimension of H
        # z_dim: dimension of Z
        # activation: function (e.g., sigmoid or relu)
        # A_adj: adjacent matrix of the graph
        super(VGAE, self).__init__()
        self.z_dim = z_dim

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
        init_range = np.sqrt(6.0/(in_dim + out_dim))
        init_param = torch.rand(in_dim, out_dim)*2*init_range - init_range
        self.W = nn.Parameter(init_param)
        self.activation = activation
       
    def forward(self, adj, h):
        # sigmoid(AHW) (A: normalized A)
        X = h @ self.W
        X = adj @ X
        if self.activation:
            x = self.activation(X)
        return X

# converts scipy sparse matrix to pytorch sparse matrix ef set_torch_type(mat):
def set_torch_type(mat):
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    coords = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    torch_mat = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return torch_mat

# Function for Edge probability calculation (public call)
# input: 
#   A_org: adjacent matrix of original graph
#   features: feature vector
#   learning_rate: learning rate given by the main function
#   n_epochs: number of iteration for training 
#   device: in here, we assume 'cpu'
# output:
#   A_ep: edge probaility matrix
#   losses: loss vector w.r.t. epoch
#   precisions: avg. precision w.r.t. epoch
def edge_probs(A_org, features, learning_rate, n_epochs, device):
    # data arrangement
    features = torch.FloatTensor(features.toarray())
    features = F.normalize(features, p=1, dim=1)
    features = features.to(device)

    assert sp.issparse(A_org)
    adj_matrix = sp.coo_matrix(A_org)

    # A'
    adj_matrix.setdiag(1)
    A_org = set_torch_type(adj_matrix).to_dense()

    # calculate A_hat
    D = np.array(adj_matrix.sum(1))
    D_inv = sp.diags(np.power(D, -0.5).flatten())
    A_hat = D_inv @ adj_matrix @ D_inv
    A_hat = set_torch_type(A_hat)

    # A = set_torch_type(A_hat)

    # sample edges for edge prediction (10% = 0.1)
    adj_matrix = sp.csr_matrix(adj_matrix)
    n_edges_sample = int(0.1 * adj_matrix.nnz / 2)
    neg_edges = []
    added_edges = set()

    # randomly select edges without duplication
    while len(neg_edges) < n_edges_sample:
        i = np.random.randint(0, adj_matrix.shape[0])
        j = np.random.randint(0, adj_matrix.shape[0])
        # diagonal,
        if i == j:
            continue
        # there is an edge, 
        if adj_matrix[i,j] > 0:
            continue
        # selected edges are already added, 
        if (i,j) in added_edges:
            continue
        neg_edges.append([i,j])
        added_edges.add((i,j))
        added_edges.add((j,i))

    neg_edges = np.asarray(neg_edges)
    nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
    np.random.shuffle(nz_upper)
    positive_edges = nz_upper[:n_edges_sample]
    # target edges to validate (check)
    val_edges = np.concatenate((positive_edges, neg_edges), axis=0)
    # original edge labels to compare
    edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)

    # Set VGAE 
    ep_net = VGAE(feature_dim=1433, h_dim=128, z_dim=32, activation=F.relu)

    # weights for training EP net
    norm_w = A_org.shape[0]**2 / float((A_org.shape[0]**2 - A_org.sum()) * 2)
    pos_weight = torch.FloatTensor([float(A_org.shape[0]**2 - A_org.sum()) / A_org.sum()]).to(device)

    # start the training
    optimizer = torch.optim.Adam(ep_net.parameters(), lr=learning_rate)
    ep_net.train()

    losses = []
    precisions = []
    # iteration for GCN
    for epoch in range(n_epochs):
        ##############
        A_pred = ep_net(A_hat, features)
        # edge probability loss
        loss = norm_w * F.binary_cross_entropy_with_logits(A_pred, A_org, pos_weight = pos_weight)
        # KL divergence 
        KL_div = 0.5 / A_pred.size(0) * (1 + 2*ep_net.log_std - ep_net.mean*2 - torch.exp(2*ep_net.log_std)).sum(1).mean()
        loss -= KL_div

        # save loss
        losses.append(loss.item())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # apply sigmoid function (after Z @ Z.T)
        A_ep = torch.sigmoid(A_pred).detach()

        # calculate precision using scipy libs. (average_precision_score function)
        est_label = A_ep[val_edges.T]
        est_label = np.nan_to_num(est_label)
        avg_precision = average_precision_score(edge_labels, est_label)

        # save precision
        precisions.append(avg_precision)
        print('loss = ', loss.item(), 'ap = ', avg_precision)

    A_re = A_ep.numpy()
    np.fill_diagonal(A_re, 0)
    return A_ep, losses, precisions



