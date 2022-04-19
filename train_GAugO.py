from models.GCN_AugO import *
import pickle
import json
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
import time

import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

if __name__ == "__main__":


    if not torch.cuda.is_available():
        gpu = -1
    else:
        gpu = 0


    tvt_nids = pickle.load(open(f'data/cora_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/cora_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/cora_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/cora_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    
    print(features)

    params_all = json.load(open('best_parameters.json', 'r'))
    params = params_all['GAugO']['cora']['gcn']

    lr =  0.01
    n_layers = 1
    accs = []
    for _ in range(50):
        start_time = time.time()
        model = AugO(adj_orig, features, labels, tvt_nids, cuda=gpu,  beta=params['beta'], temperature=params['temp'],  lr=lr, n_layers=n_layers, log=True, feat_norm='row')
        acc = model.fit(160, 30)
        accs.append(acc)
        end_time = time.time()
        
        print(f'F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')
        print('Time Eplased in Seconds: ', str(end_time-start_time))

    # pics
    plt.plot(accs, label='GAugO: F1 of each test')
    plt.xlabel('Test')  
    plt.legend()
    plt.show()
    plt.savefig('GAugO_test.png')

