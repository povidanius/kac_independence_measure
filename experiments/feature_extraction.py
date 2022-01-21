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
import sklearn.preprocessing


from uci_loader import UCIDatasets

def one_hot(x, num_classes=2):
  return np.squeeze(np.eye(num_classes)[x.reshape(-1)])


if __name__ == "__main__":

    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')

    datasets = UCIDatasets('wine', n_splits=2)



    n_batch = 256 #2048
    dim_x = datasets.in_dim # 1024
    dim_y = 1 # 32
    num_epochs = 200 #500
    num_classes = 2

    #feature_frac =     
    input_proj_dim = int(0.5*dim_x) #0

    kim = KacIndependenceMeasure(dim_x, dim_y, lr=0.005, input_projection_dim = input_proj_dim, weight_decay=0.01)
    knn = KNeighborsClassifier(n_neighbors=3)
    logistic = linear_model.LogisticRegression(max_iter=1000)
    one_hot = sklearn.preprocessing.LabelBinarizer()

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


  

    #for x,y in test_loader:
    X_train = train_data.tensors[0].detach().numpy()
    F_train = kim.project(train_data.tensors[0]).detach().numpy()
    y_train = train_data.tensors[1].detach().numpy()
    one_hot.fit(range(max(y_train)+1))
    y_train1 = one_hot.transform(y_train)


    X_test = test_data.tensors[0].detach().numpy()
    F_test = kim.project(test_data.tensors[0]).detach().numpy()
    y_test = test_data.tensors[1].detach().numpy()

    print(F_train.shape)
    print(F_test.shape)

    breakpoint()

    print("LR score(R): %f" % logistic.fit(X_train, y_train).score(X_test, y_test))
    #print("LR score(F): %f" % logistic.fit(F_train, y_train).score(F_train, y_test))
    #print("KNN score(R): %f" % knn.fit(X_train, y_train).score(X_test, y_test))
    #print("KNN score(F): %f" % knn.fit(F_train, y_train).score(F_train, y_test))


