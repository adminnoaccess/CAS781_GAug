###########################################################################
# Test edge probability calculation for preprocessing
# In the main function, there is a parameter (n_epochs and lr)
# Input: CORA dataset (saved in the pkl format)
# Output: Save 'proprocessing_edge_probability.png' (Epoch vs. loss & avg. precision)
###########################################################################
import pickle
import numpy as np
import matplotlib.pyplot as plt

from models.VGAE_edge_prob import *

def main():
    # load dataset: CORA (saved in pkl format)
    A_org = pickle.load(open(f'./data/cora_adj.pkl', 'rb'))
    features = pickle.load(open(f'./data/cora_features.pkl', 'rb'))
    labels = pickle.load(open(f'./data/cora_labels.pkl', 'rb'))

    # parameters
    n_epochs = 100
    lr = 0.01

    # Set device here 
    device = torch.device('cpu')
    A_ep, losses, precisions = edge_probs(A_org, features, lr, n_epochs, device)
    
    # Draw picture of Loss function and Avg. precision of preprocessing
    plt.plot(losses, label='Loss_ep')
    plt.plot(precisions, 'r', label='Avg. Precision')
    plt.xlabel('Epochs')  
    plt.legend()
    plt.show()
    plt.savefig('preprocessing_edge_probability.png')

if __name__ == "__main__":
    main()


