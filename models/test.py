import gc
import copy
import time
import logging
import pyro
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import collections
from sklearn.metrics import f1_score
from models.VGAE_edge_prob import *

class AugO(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, cuda=-1, hidden_size=128, emb_size=32, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, gae=False, beta=0.5, temperature=0.2, log=True, name='debug', warmup=3, gnnlayer_type='gcn', jknet=False, alpha=1,  feat_norm='row'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.gae = gae
        self.beta = beta
        self.warmup = warmup
        self.feat_norm = feat_norm
        self.features = features
        self.dropout = 0.5
        
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        # normalize feature matrix if needed
        if self.feat_norm == 'row':
            self.features = F.normalize(self.features, p=1, dim=1)
        elif self.feat_norm == 'col':
            self.features = self.col_normalization(self.features)
            
        # original adj_matrix for training vgae (torch.FloatTensor)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        self.adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()
        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        
        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)                
        self.adj = scipysp_to_pytorchsp(adj_norm)

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
                
        
        

        self.device = torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.model = GAug_model(self.features.size(1),
        #                         hidden_size,
        #                         emb_size,
        #                         self.out_size,
        #                         n_layers,
        #                         F.relu,
        #                         dropout,
        #                         self.device,
        #                         gnnlayer_type,
        #                         temperature=temperature,
        #                         gae=gae,
        #                         jknet=jknet,
        #                         alpha=alpha,
        #                         sample_type=sample_type)    
        
        # edge prediction and node classification initialization
        self.epNet = VGAE(self.features.size(1),hidden_size, emb_size, F.relu)   
        self.ncNet = GNN(self.features.size(1),
                                hidden_size,                                
                                self.out_size,
                                n_layers,
                                F.relu,self.dropout) 
        
        self.loadData(adj_matrix, features, labels, tvt_nids)
        
        print("AugO created")
            
        
    def sampleEdge(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled   
    
    def normalize_adj(self, adj):
       
        adj.fill_diagonal_(1)        
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
        adj = D_norm @ adj @ D_norm
        return adj        
    
    def forward(self, adj, adj_orig, features):
        adj_logits = self.ep_net(adj, features)

        if self.alpha == 1:
            adj_new = self.sample_adj(adj_logits)
        else:
            adj_new = self.sample_adj_add_bernoulli(adj_logits, adj_orig, self.alpha)
        adj_new_normed = self.normalize_adj(adj_new)
        nc_logits = self.nc_net(adj_new_normed, features)
        return nc_logits, adj_logits    
    
    def loadData(self, adj_matrix, features, labels, tvt_nids):
    
      

        print("AugO data loaded")       

    def pretrain_ep_net(self, model, adj, features, adj_orig, norm_w, pos_weight, n_epochs):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(model.ep_net.parameters(),
                                     lr=self.lr)
        model.train()
        for epoch in range(n_epochs):
            adj_logits = model.ep_net(adj, features)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            if not self.gae:
                mu = model.ep_net.mean
                lgstd = model.ep_net.logstd
                kl_divergence = 0.5/adj_logits.size(0) * (1 + 2*lgstd - mu**2 - torch.exp(2*lgstd)).sum(1).mean()
                loss -= kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)


    def pretrain_nc_net(self, model, adj, features, labels, n_epochs):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(model.nc_net.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.
        for epoch in range(n_epochs):
            model.train()
            nc_logits = model.nc_net(adj, features)
            # losses
            loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])


        
    def fit(self,pretrain_ep=200, pretrain_nc=20):
        
        # move data to device
        adj_norm = self.adj_norm.to(self.device)
        adj = self.adj.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        adj_orig = self.adj_orig.to(self.device)
        # model = self.model.to(self.device)
        # weights for log_lik loss when training EP net
        adj_t = self.adj_orig
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.device)
        # pretrain VGAE if needed
        if pretrain_ep:
            self.pretrain_ep_net(model, adj_norm, features, adj_orig, norm_w, pos_weight, pretrain_ep)
        # pretrain GCN if needed
        if pretrain_nc:
            self.pretrain_nc_net(model, adj, features, labels, pretrain_nc)
        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(model.ep_net.parameters(),
                                                    lr=self.lr),
                                   torch.optim.Adam(model.nc_net.parameters(),
                                                    lr=self.lr,
                                                    weight_decay=self.weight_decay))
        # get the learning rate schedule for the optimizer of ep_net if needed
        if self.warmup:
            ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.n_epochs, self.lr, self.warmup)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        patience_step = 0
        # train model
        for epoch in range(self.n_epochs):
            # update the learning rate for ep_net if needed
            if self.warmup:
                optims.update_lr(0, ep_lr_schedule[epoch])

            model.train()
            nc_logits, adj_logits = model(adj_norm, adj_orig, features)

            # losses
            loss = nc_loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            loss += self.beta * ep_loss
            optims.zero_grad()
            loss.backward()
            optims.step()
            # validate (without dropout)
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc, test_acc))
                patience_step = 0
            else:
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc))
                patience_step += 1
                if patience_step == 100:
                    self.logger.info('Early stop!')
                    break
        # get final test result without early stop
        with torch.no_grad():
            nc_logits_eval = model.nc_net(adj, features)
        test_acc_final = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
        # log both results
        self.logger.info('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        # release RAM and GPU memory
        del adj, features, labels, adj_orig
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc        
        print('AugO learning')
        
# Simple GCN layer from https://arxiv.org/abs/1609.02907
class GCNLayer(nn.Module):
    def __init__(self, in_dim, output_dim, n_heads, activation,dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)   
        
                                                                

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


class GNN(nn.Module):

    def __init__(self, dim_feats, dim_h, n_classes, n_layers, activation, dropout):
        super(GNN, self).__init__()
        heads = [1] * (n_layers + 1)
        
        
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(dim_h*heads[i], dim_h, heads[i+1], activation, dropout))
        # output layer
        self.layers.append(GCNLayer(dim_h*heads[-2], n_classes, heads[-1], None, dropout))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        return h
    
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
    def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
        """ schedule the learning rate with the sigmoid function.
        The learning rate will start with near zero and end with near lr """
        factors = torch.FloatTensor(np.arange(n_epochs))
        factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
        factors = torch.sigmoid(factors)
        # range the factors to [0, 1]
        factors = (factors - factors[0]) / (factors[-1] - factors[0])
        lr_schedule = factors * lr
        return lr_schedule

    @staticmethod
    def get_logger(name):
        """ create a nice logger """
        logger = logging.getLogger(name)
        # clear handlers if they were created in other runs
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # create console handler add add to logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler add add to logger when name is not None
        if name is not None:
            fh = logging.FileHandler(f'GAug-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def col_normalization(features):
        """ column normalization for feature matrix """
        features = features.numpy()
        m = features.mean(axis=0)
        s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
        features -= m
        features /= s
        return torch.FloatTensor(features)    
    
def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

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

    def update_lr(self, op_index, new_lr):
        """ update the learning rate of one optimizer
        Parameters: op_index: the index of the optimizer to update
                    new_lr:   new learning rate for that optimizer """
        for param_group in self.optimizers[op_index].param_groups:
            param_group['lr'] = new_lr