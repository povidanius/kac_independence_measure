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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn import datasets, neighbors, linear_model
import sklearn.preprocessing

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.datasets import fetch_openml
import pandas as pd

from kac_independence_measure import KacIndependenceMeasure


def one_hot(x, num_classes=2):
  return np.squeeze(np.eye(num_classes)[x.reshape(-1)])

def load_data(train_frac):
    #eeg-eye-state
    X,y = fetch_openml(name="ionosphere", as_frame=True, return_X_y=True)
    print(X.shape)
    #X = X.to_numpy()[:,2:]
    #breakpoint()
    dim_x = X.shape[1]
    categories = pd.unique(y.to_numpy().ravel())
    y_num = np.zeros(len(y))
    category_id = 0
    for cat in categories:
        ind = np.where(y == cat)
        #breakpoint()
        for ii in ind:
            y_num[ii] = category_id
        category_id = category_id + 1

    y = y_num.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, train_size=int(train_frac*X.shape[0])) 
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')


    n_batch = 1024 #2048
    num_epochs = 300 #500
    num_classes = 2
    dim_x = 34 # 1024
    dim_y = num_classes # 32
    train_frac = 0.5
    normalize = False

    #feature_frac =     
    input_proj_dim = int(0.5*dim_x) #0

    X_train, y_train, X_test, y_test = load_data(train_frac)

    kim = KacIndependenceMeasure(dim_x, dim_y, lr=0.007, input_projection_dim = input_proj_dim, weight_decay=0.01)
    knn = KNeighborsClassifier(n_neighbors=3)
    logistic = linear_model.LogisticRegression(max_iter=1000)
    svc = SVC(kernel='poly', gamma='auto', degree=2)
    lsvm = LinearSVC(random_state=42, max_iter=10000)
    #one_hot = sklearn.preprocessing.LabelBinarizer()


    n = X_train.shape[0]
    ytr = one_hot(y_train, num_classes)
    yte = one_hot(y_test, num_classes)
    dep_history = []
    for i in range(num_epochs):
        print("Epoch: {}".format(i))
        shuffled_indices = np.arange(n)
        np.random.shuffle(shuffled_indices)
        num_batches = math.ceil(n/n_batch)
        for j in np.arange(num_batches):
            batch_indices = shuffled_indices[n_batch*j:n_batch*(j+1)]
            Xb = torch.from_numpy(X_train[batch_indices, :].astype(np.float32))
            yb = torch.from_numpy(ytr[batch_indices, :].astype(np.float32)) #.unsqueeze(1)
            #print("{} {}".format(Xb.shape, yb.shape))
            #breakpoint()
            dep = kim.forward(Xb, yb, normalize=normalize)
            dep_history.append(dep.detach().numpy())
            print("{} {} {}, {}".format(i, j, dep, Xb.shape[0]))

  

    F_train = kim.project(torch.from_numpy(X_train.astype(np.float32)), normalize=normalize).detach().numpy()
    #X_test = test_data.tensors[0].detach().numpy()
    F_test = kim.project(torch.from_numpy(X_test.astype(np.float32)), normalize=normalize).detach().numpy() #test_data.tensors[0]).detach().numpy()
    #y_test = test_data.tensors[1].detach().numpy()
    
    print(X_train.shape)
    print(X_test.shape)
    print(F_train.shape)
    print(F_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    plt.plot(dep_history)
    plt.show()
    #breakpoint()

    print("LR score(R): %f" % logistic.fit(X_train, y_train).score(X_test, y_test))
    print("LR score(F): %f" % logistic.fit(F_train, y_train).score(F_test, y_test))
    print("KNN score(R): %f" % knn.fit(X_train, y_train).score(X_test, y_test))
    print("KNN score(F): %f" % knn.fit(F_train, y_train).score(F_test, y_test))
    print("LSVM score(R): %f" % lsvm.fit(X_train, y_train).score(X_test, y_test))
    print("LSVM score(F): %f" % lsvm.fit(F_train, y_train).score(F_test, y_test))
    print("SVM score(R): %f" % svc.fit(X_train, y_train).score(X_test, y_test))
    print("SVM score(F): %f" % svc.fit(F_train, y_train).score(F_test, y_test))


