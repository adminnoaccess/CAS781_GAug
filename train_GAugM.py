import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from models.GCN_AugM import *
from models.VGAE_edge_prob import *
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'cora')
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--gpu', type=str, default = '0')
parser.add_argument('--i', type=str, default = '2')
parser.add_argument('--lr', type=int, default = 0.01)
args = parser.parse_args()

if __name__ == "__main__":

    if args.gpu == '-1':
        gpu_avail = -1
    else:
        gpu_avail = 0
    # config device
    # if torch.cuda.is_available():
    #     print('GPU available?', torch.cuda.is_available())
    #     if gpu_avail == -1:
    #         print ('You are using CPU')
    #         device = torch.device('cpu')
    #     elif gpu_avail == 0:
    #         print ('You are using GPU')
    #         device = torch.device('cuda')
    # else:
    #     print ('You are using CPU')
    #     device = torch.device('cpu')

    tvt_nids = pickle.load(open(f'data/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/{args.dataset}_labels.pkl', 'rb'))


    best_params = json.load(open('best_parameters.json', 'r'))
    params = best_params['GAugM'][args.dataset]['gcn']

    # i = 5 for GAugM cora gcn
    i = params['i']
    # Edge probability
    # A_pred = pickle.load(open(f'data/GAugM_edge_probabilities/{args.dataset}_graph_{i}_logits.pkl', 'rb'))
    print('Training edge probabilities on cpu...')
    A_pred,_ ,_ = edge_probs(A_org = adj_orig, features = features, learning_rate = 0.01, n_epochs = 200, device=torch.device('cpu'))
    
    # modify the original adj w.r.t edge prediction
    adj = AugM_GCN.pred_adj(adj_orig = adj_orig, A_pred = A_pred, remove_pct = params['rm_pct'], add_pct = params['add_pct'])
    
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    accs = []
    best_vali_accs = []
    times = []
    # best_logits = []
    counter = 0 #
    for _ in range(30):
        gnn = AugM_GCN(adj, A_pred, features, labels, tvt_nids, rm = params['rm_pct'], add = params['add_pct'], cuda = gpu_avail, epochs = args.epochs, lr = args.lr)
        acc, best_vali_acc, best_logit, time = gnn.fit()
        accs.append(acc)
        best_vali_accs.append(best_vali_acc)
        times.append(time)
        # best_logits.append(best_logit)
        counter += 1 
        print (f'Range {counter} finished')
        # print(times)
    print(f'\nF1_micro: mean test acc: {np.mean(accs):.4f}, std test acc: {np.std(accs):.4f}, best vali acc: {np.max(best_vali_accs):.4f}\ntotal time for this run: {np.sum(times):.3f}s, average time for every {args.epochs} epochs: {np.mean(times):.3f}s')
    
    # pics
    plt.plot(accs, label='GAugM: F1 of each test')
    plt.xlabel('Test')  
    plt.legend()
    plt.show()
    plt.savefig('GAugM_test.png')
