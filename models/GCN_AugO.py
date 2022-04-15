import gc
import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from itertools import combinations
from sklearn.metrics import roc_auc_score, average_precision_score


from models.VGAE_edge_prob import *



def sp2tensor(sp_matrix):

    if not sp.isspmatrix_coo(sp_matrix):
        sp_matrix = sp_matrix.tocoo()
        
    # print(type(sp_matrixs))
    coords = np.vstack((sp_matrix.row, sp_matrix.col)).transpose()
    values = sp_matrix.data
    shape = sp_matrix.shape
    
    result = torch.sparse.FloatTensor(torch.LongTensor(coords.T),torch.FloatTensor(values),torch.Size(shape))
    return result       


class AugO(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, cuda=-1, hidden_size=128, emb_size=32, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, beta=0.5, temperature=0.2, log=True, name='debug',  gnnlayer_type='gcn', jknet=False,  feat_norm='row'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.beta = beta
        self.feat_norm = feat_norm
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        if seed>0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
        self.load_data(adj_matrix, features, labels, tvt_nids)
        self.model = Model(self.features.size(1),
                                hidden_size,
                                emb_size,
                                self.out_size,
                                n_layers,
                                F.relu,
                                dropout,
                                self.device,
                                temperature=temperature
                                )
        

    def load_data(self, adj_matrix, features, labels, tvt_nids):


        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)            
        if self.feat_norm == 'row':
            self.features = F.normalize(self.features, p=1, dim=1)
        elif self.feat_norm == 'col':
            self.features = self.col_normalization(self.features)
        
        # preprocessing
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        self.adj_original = sp2tensor(adj_matrix).to_dense()
        
        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        self.adj_norm = sp2tensor(adj_norm)
        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        
        self.adj = sp2tensor(adj_norm)

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
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)
        # sample the edges to evaluate edge prediction results
        # sample 10% (1% for large graph) of the edges and the same number of no-edges
        if labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj_matrix = sp.csr_matrix(adj_matrix)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)
        # sample negative edges
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j:
                continue
            if adj_matrix[i, j] > 0:
                continue
            if (i, j) in added_edges:
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)
        # sample positive edges
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)

    def trainEdgeNet(self, model, adj, features, adj_original, norm_w, pos_weight, n_epochs):

        optimizer = torch.optim.Adam(model.edgePredNet.parameters(),lr=self.lr)
        
        for epoch in range(n_epochs):
            model.train()  
            adj_logits = model.edgePredNet(adj, features)
            print(torch.sum(adj_logits))
                      
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_original, pos_weight=pos_weight)   
            optimizer.zero_grad()        
            loss.backward()        
            optimizer.step()
            
            model.eval()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()            
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            print('EPNet pretrain, Epoch [{:3}/{}]: loss {:.4f}, auc {:.4f}, ap {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), ep_auc, ep_ap))

    def trainNodeNet(self, model, adj, features, labels, n_epochs):
        
        optimizer = torch.optim.Adam(model.nodeClassifNet.parameters(),lr=self.lr,weight_decay=self.weight_decay)    
        
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        
        max_acc = 0.
        for epoch in range(n_epochs):
            
            model.train()
            nc_logits = model.nodeClassifNet(adj, features)            
            
            loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nodeClassifNet(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            if val_acc > max_acc:
                max_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                print('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc, test_acc))
            else:
                print('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc))
            
        
    # train the AugO model  
    def fit(self, pretrain_ep=200, pretrain_nc=20):
        
        # move data to device
        adj_norm = self.adj_norm.to(self.device)
        adj = self.adj.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        model = self.model.to(self.device)        
        adj_original = self.adj_original.to(self.device)
        
        # initialize the adj and weights
        adj_t = self.adj_original
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.device)
 
        # initialize and train the node classification and the edge predictor
        self.trainEdgeNet(model, adj_norm, features, adj_original, norm_w, pos_weight, pretrain_ep)
        self.trainNodeNet(model, adj, features, labels, pretrain_nc)
        
        # optimizers for both the edge and the node classfier
        optimzers = MultipleOptimizer(torch.optim.Adam(model.edgePredNet.parameters(),
                                                    lr=self.lr),
                                   torch.optim.Adam(model.nodeClassifNet.parameters(),
                                                    lr=self.lr,
                                                    weight_decay=self.weight_decay))
  

        # loss function for node classification as defined in the paper
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
   
        
        
        # initialize the counters before the training
        max_acc = 0.
        earlystop_counter = 0
        
        # train model
        for i in range(self.n_epochs):
            print('epoch '+str(i+1)+'/'+str(self.n_epochs))


            # forward function 
            model.train()
            nc_logits, adj_logits = model(adj_norm, adj_original, features)

            # losses
            loss = nc_loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_original, pos_weight=pos_weight)
            loss += self.beta * ep_loss
            
            # using the group optimizer for both the node and edge classifier
            optimzers.zero_grad()
            loss.backward()
            optimzers.step()
            
            # evaluation
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nodeClassifNet(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            if val_acc > max_acc:
                
                earlystop_counter = 0
                max_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                print('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(i+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc, test_acc))
                
            else:
                # early stop if the best result can be 
                print('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}'
                            .format(i+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc))
                earlystop_counter += 1
                if earlystop_counter == 50:
                    print("Early stop, exit current epoch")
                    break
                
        # get final test result without early stop
        with torch.no_grad():
            nc_logits_eval = model.nodeClassifNet(adj, features)
        test_acc_final = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])

        # log both results
        print('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        # release RAM and GPU memory
        del adj, features, labels, adj_original
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc



    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        logits = adj_pred[val_edges.T]
        logits = np.nan_to_num(logits)
        roc_auc = roc_auc_score(edge_labels, logits)
        ap_score = average_precision_score(edge_labels, logits)
        return roc_auc, ap_score
    
     
    

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        """ evaluate node classification results """
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(nc_logits))
            tp = len(torch.nonzero(preds * labels))
            tn = len(torch.nonzero((1-preds) * (1-labels)))
            fp = len(torch.nonzero(preds * (1-labels)))
            fn = len(torch.nonzero((1-preds) * labels))
            pre, rec, f1 = 0., 0., 0.
            if tp+fp > 0:
                pre = tp / (tp + fp)
            if tp+fn > 0:
                rec = tp / (tp + fn)
            if pre+rec > 0:
                fmeasure = (2 * pre * rec) / (pre + rec)
        else:
            preds = torch.argmax(nc_logits, dim=1)
            correct = torch.sum(preds == labels)
            fmeasure = correct.item() / len(labels)
        return fmeasure
    





    @staticmethod
    def col_normalization(features):
        
        features = features.numpy()        
        meanval = features.mean(axis=0)
        # add small value to make sure its non 0
        stddevi = features.std(axis=0, ddof=0, keepdims=True) + 1e-11
        features = (features-meanval)/stddevi
        
        return torch.FloatTensor(features)




# the base model structure for AugO model
class Model(nn.Module):
    def __init__(self,
                 dim_features,
                 dim_h,
                 dim_z,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 device,        
                 temperature=1
                 ):
        super(Model, self).__init__()
        self.device = device
        self.temperature = temperature


        print(activation)
        self.edgePredNet = VGAE(dim_features, dim_h, dim_z, activation)
        self.nodeClassifNet = GNN(dim_features, dim_h, n_classes, n_layers, activation, dropout)


    def sample_adj_matrix(self, adj_logits):
        
        

        adj_probs = adj_logits / torch.max(adj_logits)
        # print(torch.sum(adj_probs))
        sampled_raw = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=adj_probs).rsample()

        temp = sampled_raw.triu(1)
        result = temp + temp.T
        
        # normialization
        result.fill_diagonal_(1)
        normialized = torch.diag(torch.pow(result.sum(1), -0.5)).to(self.device)
        result = normialized @ result @ normialized
        
        return result   




    def forward(self, adj, adj_original, features):
        

        adj_logits = self.edgePredNet(adj, features)
        # print("before")
        print(torch.sum(adj_logits))
        adj_new = self.sample_adj_matrix(adj_logits)    
        # print("after")
            
        nc_logits = self.nodeClassifNet(adj_new, features)
        return nc_logits, adj_logits


# Make GCN Layer for NN
class GCNLayer(nn.Module):  # single layer of GCN
    def __init__(self, in_dim, out_dim, n_heads, activation, dropout):
        super(GCNLayer, self).__init__()
        # Set weight matrix (W)
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        init_range = np.sqrt(6.0/(in_dim + out_dim))
        init_param = torch.rand(in_dim, out_dim)*2*init_range - init_range
        self.W = nn.Parameter(init_param)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
                        
       
    def forward(self, adj, h):

        if self.dropout:
            h = self.dropout(h)        
        
        X = h @ self.W
        X = adj @ X
        if self.activation:
            X = self.activation(X)
            
        return X


class GNN(nn.Module):
    # model for node classigication
    
    def __init__(self, dim_features, dim_h, n_classes, n_layers, activation, dropout):
        super(GNN, self).__init__()
        heads = [1] * (n_layers + 1)
        
        # construct the layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(dim_features, dim_h, heads[0], activation, 0))
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(dim_h*heads[i], dim_h, heads[i+1], activation, dropout))
        self.layers.append(GCNLayer(dim_h*heads[-2], n_classes, heads[-1], None, dropout))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        return h




class VGAE(nn.Module):  # edge probability calculation using GCN
    def __init__(self, feature_dim, h_dim, z_dim, activation):
        # feaature_dim: dimension of feature matrix
        # h_dim: dimension of H
        # z_dim: dimension of Z
        # activation: function (e.g., sigmoid or relu)
        # dropout for different layers, as input and output needs some different settings
        # A_adj: adjacent matrix of the graph
        super(VGAE, self).__init__()

        # make GCN Layer 
        self.gcn_base = GCNLayer(feature_dim, h_dim, 1, None, 0 )
        self.gcn_mean = GCNLayer(h_dim, z_dim, 1, F.relu, 0)
        self.gcn_log_std = GCNLayer(h_dim, z_dim, 1, activation,0 )
        

    def forward(self, A_adj, features):
        # A_adj: adjacent matrix of the graph
        # features: feature matrix
        # GCN encoder
        hidden_layer = self.gcn_base(A_adj, features)
        self.mean = self.gcn_mean(A_adj, hidden_layer)
        Z = self.mean

        # decode
        # (inner product) and return edge probablity
        A_pred = Z @ Z.T
        return A_pred







class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()











