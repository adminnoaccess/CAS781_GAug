import gc
import math
import copy
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import f1_score

class GCN(object):
    def __init__(self, adj_orig, A_pred, features, labels, tvt_nids, add, rm, cuda=-1, hidden_size=128, num_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, print_progress=True, dropedge=0):
        self.t = time.time()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.print_progress = print_progress
        self.dropedge = dropedge

        # config device
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda>=0 else 'cpu')
        # fix random seeds if needed
        # if seed > 0:
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)

        adj = self.sample_graph_det(adj_orig, A_pred, rm, add)
        adj_eval = adj
        # load the data
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        #create GNN(GCN here) model
        self.model = GCN_model(self.features.size(1), # in_feats, from load_data
                               hidden_size, # hidden_size 
                               self.n_classes, # n_classes, from load_data
                               num_layers, # n_layers
                               F.relu, #activation  
                               dropout) 
        # move everything to device
        self.model.to(self.device)

    # Apply prediction to the original adj matrix   
    def sample_graph_det(self, adj_orig, A_pred, remove_pct, add_pct):
        # if no edges are removed or added
        if remove_pct == 0 and add_pct == 0:
            return copy.deepcopy(adj_orig)
    
        # the upper triangular portion of a adj_orig in sparse format
        orig_upper = sp.triu(adj_orig, k = 1)
        # number of nonzeros values in the upper triangular portion of the adj_orig
        n_edges = orig_upper.nnz
        # the indexes of nonzeros
        edges = np.transpose(np.nonzero(orig_upper))

        if remove_pct:
            n_remove = int(n_edges * remove_pct / 100)
            # nonzeros(edges)' indices in edge probability matrix
            pos_probs = A_pred[edges.T[0], edges.T[1]]
            # index of top n_remove items in pos_probs
            index_of_removed = np.argpartition(pos_probs, n_remove)[:n_remove]
            mask = np.ones(len(edges), dtype = bool)
            mask[index_of_removed] = False
            edges_pred = edges[mask]
        else:
            edges_pred = edges

        if add_pct:
            n_add = int(n_edges * add_pct / 100)
            # deep copy to avoid modifying A_pred
            A_probs = copy.deepcopy(A_pred)
            # make the probabilities of the lower half to be zero (including diagonal)
            A_probs = np.triu(A_probs, k = 1)
            # A_probs[np.tril_indices(A_probs.shape[0])] = 0
            # make the probabilities of existing edges to be zero
            A_probs[edges.T[0], edges.T[1]] = 0
            all_probs = A_probs.reshape(-1)
            # index of top n_add items in all_probs 
            index_of_added = np.argpartition(all_probs, -n_add)[-n_add:]
            new_edges = []
            for index in index_of_added:
                i = int(index / A_probs.shape[0])
                j = index % A_probs.shape[0]
                new_edges.append([i, j])
            edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
        else:
            edges_pred = edges

        adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
        adj_pred = adj_pred + adj_pred.T
        return adj_pred

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        # Transform features into floatTensor format
        self.features = torch.FloatTensor(features)      
        # row normalization
        self.features = F.normalize(self.features, p=2, dim=1)

        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # number of classes
        if len(self.labels.size()) == 1:
            self.n_classes = len(torch.unique(self.labels))
        else:
            self.n_classes = labels.size(1)

        # adj for training
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj
        adj = sp.csr_matrix(adj)
        self.G = DGLGraph(self.adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

        # adj for inference      
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        self.G_eval = DGLGraph(self.adj_eval)
        # normalization (D^{-1/2})
        degs_eval = self.G_eval.in_degrees().float()
        norm_eval = torch.pow(degs_eval, -0.5)
        norm_eval[torch.isinf(norm_eval)] = 0
        norm_eval = norm_eval.to(self.device)
        self.G_eval.ndata['norm'] = norm_eval.unsqueeze(1)

    def dropEdge(self):
        upper = sp.triu(self.adj, 1)
        n_edge = upper.nnz
        n_edge_left = int((1 - self.dropedge) * n_edge)
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        adj = adj + adj.T
        adj.setdiag(1)
        self.G = DGLGraph(adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

    #for training and testing
    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            if self.dropedge > 0:
                self.dropEdge()
            self.model.train()
            logits = self.model(self.G, features)
            # losses
            # l = F.nll_loss(logits[self.train_nid], labels[self.train_nid])
            l = nc_criterion(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # validate with original graph (without dropout)
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(self.G_eval, features).detach().cpu()
            vali_acc, _ = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid].cpu())
            if self.print_progress:
                print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, self.epochs, l.item(), vali_acc))
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                test_acc, conf_mat = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
                if self.print_progress:
                    print(f'                 test acc: {test_acc:.4f}')
        if self.print_progress:
            print(f'Final test results: acc: {test_acc:.4f}')
        del self.model, features, labels, self.G
        torch.cuda.empty_cache()
        gc.collect()
        t = time.time() - self.t
        return test_acc, best_vali_acc, best_logits

    def eval_node_cls(self, logits, labels):
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        return micro_f1, 1

# Simple GCN layer from https://arxiv.org/abs/1609.02907
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim)) # dim_bias = dim_out
        else:
            self.register_parameter('bias', None)

        #initialize the parameters 
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.001)                                                           

    def forward(self, adj, h):
        # if self.dropout:
        #     h = self.dropout(h)
        h = torch.mm(h, self.weight)

        # normalization by square root of src degree
        h = h * adj.ndata['norm']
        adj.ndata['h'] = h
        adj.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        h = adj.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * adj.ndata['norm']
        
        # bias
        if self.bias is not None:
            h = h + self.bias
        return h

class GCN_model(nn.Module):
    def __init__(self, in_features, hidden_size, n_classes, num_hidlayers, activation, dropout):
        super(GCN_model, self).__init__()
        self.activation = activation

        self.gc1 = GCNLayer(in_features, hidden_size, activation, dropout)
        self.gc2 = GCNLayer(hidden_size, n_classes, activation, dropout) 
        self.dropout = dropout 

    def forward(self, adj, h):
        h = self.activation(self.gc1(adj, h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(adj, h)
        return h  

# # edge_predictor model
# class VGAE_model(nn.Module):
#     def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
#         super(VGAE_model, self).__init__()
#         self.gae = gae
#         self.gcn_input = GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False)
#         self.gcn_hidden = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)
# #        self.gcn_logstd = GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False)

#     def forward(self, adj, features):
#         hidden = self.gcn_input(adj, features)
#         self.mean = self.gcn_hidden(adj, hidden)
#         self.logstd = self.gcn_hidden(adj, hidden)
#         if self.gae: # GAE model without sampling
#             # GAE (no sampling at bottleneck)
#             Z = self.mean
#         else: # VGAE with sampling                        
#             Noise = torch.randn_like(self.mean)*torch.exp(self.logstd)
#             Z = Noise + self.mean
#         # inner product decoder
#         adj_logits = Z @ Z.T
#         return adj_logits

