##############################################################################################
# McMaster Computing and Software Dept. 
# CAS 781 Class
# Team name: JLY
# Member: 
# Jongjun Park: VGAE
# Liuyin Shi: GAugO with GCN networks
# Yihang Xu: GAigM with GCN networks
# Paper: Data Augmentation for Graph Neural Networks (https://arxiv.org/pdf/2006.06830.pdf)
##############################################################################################

##############################################################################################
## Dataset
The dataset is saved in 'pickle' python binary type. 
##############################################################################################

##############################################################################################
## model
##############################################################################################
# 1. model/VGAE_edge_prob.py
 
 Description: Before applying GAugM or GAugO, we need to calculate the edge probability of
	original graph. In this paper, they used Variational Graph Auto-Encoder (VGAE) to 
	calculate the edge probability. VGAE simply adapted Graph Convolutional Networks (GCN)
	with single layer. 

# (1) VGAE / GCNLayer class
  Using PyTorch, we developed VGAE class with GCNLayer class. It is a simple GCN networks 
 which is described exactly at the paper (Kipf, T. N. and Welling, M. 'Variational graph autoencoders')
To get a appropriate edge probability, we need to train the VGAE before applying the algorithm.

# (2) edge_prob
 Input: A_org (adjacent matrix), features (vector), learning_rate, n_epochs, device
 Output: A_ep (edge probability matrix), losses, precisions (at each iteration)
 Description: After reading the dataset, we apply them to train the VGAE. So, the edge_prob()
 	gets an adjacent matrix, feature vector as an input, and return the edge probability 
	as an matrix format. learning_rate and number of iteration (n_epochs) could be set as 
	a user-defined parameters. If a computer support the 'cuda' you can use it with 
	setting the device as 'cuda'. To draw the losses and average precision on each iteration,
	this function also returns them. Using the example, you can draw the statistics. 

# 2. model/GCN_AugO.py
 
 Description: This is the main file for the AugO model. We use a master class GAugO class, which encapsulate the actual AugO model according to the pytroch programming practice. Within the Model class,
we also have 2 separate node classfication network and edge prediction network.

# (1) VGAE / GCNLayer class
  Using PyTorch, we developed VGAE class with GCNLayer class. It is a simple GCN networks 
 which is described exactly at the paper (Kipf, T. N. and Welling, M. 'Variational graph autoencoders')
 To get a appropriate edge probability, we need to train the VGAE before applying the algorithm.

# (2) Model class
 Description: this is the main model of the AugO. It utilize the VAGE encoder as shown above for edge
 prediction, and GCN for node classfication. The loss are fed to the GCN for learning the graph and 
 utilized for further learning

# (3) AugO class
 Describption: this is the master class for encapsulating the Model class accoriding to the pytorch 
 practice. Here all the model parameteres are intialized and the model workflow is constructed

 
##############################################################################################
## Test file
##############################################################################################
# 1. test_ep.py

 Description: Simply, this code load the 'CORA' dataset (you can change the file name to load
	different dataset. In here, we set n_epochs = 100 and learning_rate = 0.01. To run the 
	code (you can type on the terminal $ python3 test_ep.py), you can get a graph of losses 
	and average precision for preprocessing training (preprocessing_edge_probability.png). 

# 2.train_GAugO.py

 Description: Simply, this code load the 'CORA' dataset (you can change the file name to load
	different dataset. we set the number of epochs to be 50, and for each epoch, the model (GCN_AugO.py)
will also train the node classification and the edge prediction networks within


