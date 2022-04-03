import os
import copy
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from models.GCN_AugM import GCN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--i', type=str, default='2')
    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    best_params = json.load(open('best_parameters.json', 'r'))
    params = best_params['GAugM'][args.dataset]['gcn']
    # i = 5 for GAugM cora gcn
    i = params['i']
    A_pred = pickle.load(open(f'data/edge_probabilities/{args.dataset}_graph_{i}_logits.pkl', 'rb'))


    accs = []
    best_vali_accs = []
    # best_logits = []
    counter = 0 #
    for _ in range(10):
        gnn = GCN(adj_orig, A_pred, features, labels, tvt_nids, rm = params['rm_pct'], add = params['add_pct'], print_progress=True, cuda=gpu, epochs=20)
        acc, best_vali_acc, best_logit = gnn.fit()
        accs.append(acc)
        best_vali_accs.append(best_vali_acc)
        # best_logits.append(best_logit)
        counter += 1 
        print (f'Range {counter} finished')
    print(f'\nMicro F1: mean test acc: {np.mean(accs):.6f}, std test acc: {np.std(accs):.6f}, best vali acc: {np.max(best_vali_accs):.6f}')
