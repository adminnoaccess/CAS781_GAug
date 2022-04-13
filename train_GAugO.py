from models.GCN_AugO import *
import gc
import copy
import time
import pickle
import argparse
import os
import json
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import collections
from sklearn.metrics import f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    tvt_nids = pickle.load(open(f'data/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    
    print(features)

    params_all = json.load(open('best_parameters.json', 'r'))
    params = params_all['GAugO'][args.dataset][args.gnn]

    gnn = args.gnn
    layer_type = args.gnn
    feat_norm = 'row'
    lr =  0.01
    n_layers = 1
    accs = []
    for _ in range(50):
        model = AugO(adj_orig, features, labels, tvt_nids, cuda=gpu,  beta=params['beta'], temperature=params['temp'], warmup=0,  lr=lr, n_layers=n_layers, log=True, feat_norm=feat_norm)
        acc = model.fit(160, 30)
        accs.append(acc)
    print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')

