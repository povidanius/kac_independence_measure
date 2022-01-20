import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import itertools
from torch.nn.utils import weight_norm
import numpy as np
import sys
sys.path.insert(0, "../")
from scipy.signal import medfilt as mf
import os
from sklearn.utils import gen_batches

#from skorch import NeuralNetworkClassifier


from kac_independence_measure import KacIndependenceMeasure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn import datasets, neighbors, linear_model
from uci_loader import UCIDatasets

if __name__ == "__main__":

    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')

    datasets = UCIDatasets('energy', n_splits=2)



    n_batch = 256 #2048
    dim_x = datasets.in_dim # 1024
    dim_y = 1 # 32
    num_epochs = 200 #500

    #feature_frac =     
    input_proj_dim = int(0.5*dim_x) #0

    kim = KacIndependenceMeasure(dim_x, dim_y, lr=0.005, input_projection_dim = input_proj_dim, weight_decay=0.01)
    knn = KNeighborsClassifier(n_neighbors=3)
    logistic = linear_model.LogisticRegression(max_iter=1000)

    train_data = datasets.get_split(0)
    test_data = datasets.get_split(1)
    print(train_data)
    print(test_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=n_batch, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=0)

    for epoch in np.arange(num_epochs):
        for x,y in train_loader:
            #print("{} {}".format(x.shape, y.shape))
            dep = kim.forward(x,y)
            print("Epoch: {}, kim: {}".format(epoch, dep))

    

    for x,y in test_loader:
            proj = kim.project(x)
            print(proj.shape)

