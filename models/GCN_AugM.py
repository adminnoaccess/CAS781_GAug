import gc
import copy
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import collections
from sklearn.metrics import f1_score

class GCN(object):
    def __init__(self, adj_orig, A_pred, features, labels, tvt_nids, add, rm, cuda, hidden_size=128, num_layers=1, epochs=200, seed=-1, lr=0.01, weight_decay=5e-4, dropout=0.5, print_progress=True):
        self.t = time.time()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.print_progress = print_progress

        # config device
        if torch.cuda.is_available():
            print('GPU available?', torch.cuda.is_available())
            if cuda == -1:
                print ('You are using CPU')
                self.device = torch.device('cpu')
            elif cuda == 0:
                print ('You are using GPU')
                self.device = torch.device('cuda')
        else:
            print ('You are using CPU')
            self.device = torch.device('cpu')
        
        # modify the input adj matrix
        adj = self.pred_adj(adj_orig, A_pred, rm, add)
        # load the data
        self.load_data(adj, features, labels, tvt_nids)

        #create GCN model
        self.model = GCN_model(self.features.size(1), # in_feats, from load_data
                               hidden_size, # hidden_size 
                               self.n_classes, # n_classes, from load_data
                               num_layers, # n_layers
                               F.relu, #activation  
                               dropout) 
        # move everything to device
        self.model.to(self.device)

    # Apply prediction to the original adj matrix   
    def pred_adj(self, adj_orig, A_pred, remove_pct, add_pct):
        # if no edges are removed or added
        if remove_pct == 0 and add_pct == 0:
            return copy.deepcopy(adj_orig)
        # #the upper triangular portion of a adj_orig in sparse format
        orig_upper = sp.triu(adj_orig, k = 1)       
        # number of nonzeros values in the upper triangular portion of the adj_orig
        n_edges = orig_upper.nnz # 5278
        # the list of indices of nonzeros 
        edges = np.transpose(np.nonzero(orig_upper))
        # print(np.nonzero(orig_upper))
        # print(edges) # 5278
        # print(len(A_pred)) # 2708
        if remove_pct:
            n_remove = int(n_edges * remove_pct / 100)
            # nonzeros(existing edges)' indices' corresponding probabilities in edge probability matrix
            existed_edge_probs = A_pred[edges.T[0], edges.T[1]]
            # print(A_pred)
            # print(existed_edge_probs) # [0.9036223  0.9345134  0.9557134  ... 0.9410974  0.9351894  0.91263455]
            # print(len(existed_edge_probs)) # 5278            
            # # indices of top n_remove small values in pos_probs, sort among existing edges' probabilities
            indices_of_removed = np.argpartition(existed_edge_probs, n_remove)[:n_remove] #[  96  387 1601 ... 2573 5213 3088]           
            # print(len(indices_of_removed)) # 105
            # # filter = [True, True, True, ....] , length = 5278
            filter = np.ones(n_edges, dtype = bool)
            # # filter[[  96  387 1601 ... 2573 5213 3088] ] = False
            filter[indices_of_removed] = False
            # # edges with indices same with False in filter are removed 
            edges_pred = edges[filter]
        else:
            edges_pred = edges

        if add_pct:
            n_add = int(n_edges * add_pct / 100)
            # deep copy to avoid modifying A_pred
            A_pred_add = copy.deepcopy(A_pred) # 2708*2708
            # counter = np.count_nonzero(A_pred_add == 1)
            # make the probabilities of the lower half to be zero (including diagonal)
            A_pred_add = np.triu(A_pred_add, k = 1)
            # make the probabilities of existing edges to be zero
            A_pred_add[edges.T[0], edges.T[1]] = 0
            # flatten A_pred_add
            all_probs = A_pred_add.reshape(-1)
            # print(len(all_probs)) # 2708^2 = 733264
            # list of indices of top n_add large values in all_probs , sort among all edges' probabilities (existing edges' probabilities are reset to 0)
            indices_of_added = np.argpartition(all_probs, -n_add)[-n_add:] # [ 378191 1624459  873677 ... 3004429 6821415 1241286]
            # print(len(indices_of_added)) # 3008

            # filter = np.zeros(all_probs, dtype = bool)
            # # filter[[ 378191 1624459  873677 ... 3004429 6821415 1241286]] = True
            # filter[indices_of_added] = True
            # # edges with indices same with True in filter are added 
            # edges_pred = edges[filter]       

            ## add the new edges' indices
            new_edges = []
            for index in indices_of_added:
                i = int(index / A_pred_add.shape[0])
                j = index % A_pred_add.shape[0]
                new_edges.append([i, j])
            edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
        else:
            edges_pred = edges

        adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
        adj_pred = adj_pred + adj_pred.T
        return adj_pred

    def load_data(self, adj, features, labels, tvt_nids):
        # Transform features into floatTensor format
        self.features = torch.FloatTensor(features)      
        # row normalization
        self.features = F.normalize(self.features, p=2, dim=1)
        adj_eval = adj
        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        #load node ids
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
        adj = sp.csr_matrix(adj)
        self.adj = adj
        # Create a graph from a sp.sparse matrix
        self.G = dgl.from_scipy(self.adj)
        # move the graph to device
        self.G = self.G.to(self.device)
        # normalization (D^-0.5)
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

        # adj for evaluation     
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        # Create a graph from a sp.sparse matrix
        self.G_eval = dgl.from_scipy(self.adj_eval)
        # move the graph to device
        self.G_eval = self.G_eval.to(self.device)
        # normalization (D^-0.5)
        degs_eval = self.G_eval.in_degrees().float()
        norm_eval = torch.pow(degs_eval, -0.5)
        norm_eval[torch.isinf(norm_eval)] = 0
        norm_eval = norm_eval.to(self.device)
        self.G_eval.ndata['norm'] = norm_eval.unsqueeze(1)

    #for training and validating
    def fit(self):
        # define optimizer and 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        nc_criterion = nn.CrossEntropyLoss()
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # initialize the best 
        best_vali_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            # set the model in train mode
            self.model.train()
            logits = self.model(self.G, features)
            # losses
            l = nc_criterion(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # validate with original graph
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(self.G_eval, features).detach().cpu()
            vali_acc = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid].cpu())
            if self.print_progress:
                print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, self.epochs, l.item(), vali_acc))
            # when new best validate accuracy appears for validate nodes, evaluate the test nodes  
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                test_acc = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
                if self.print_progress:
                    print(f'    Best validation updated, test acc: {test_acc:.4f}')
        if self.print_progress:
            print(f'Final test results: acc: {test_acc:.4f}')
        del self.model, features, labels, self.G
        torch.cuda.empty_cache()
        gc.collect()
        t = time.time() - self.t
        return test_acc, best_vali_acc, best_logits, t

    # evaluate the accuracy with micro_f1
    def eval_node_cls(self, logits, labels):
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        return micro_f1

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
