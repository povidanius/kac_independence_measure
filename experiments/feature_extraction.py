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
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kac_independence_measure import KacIndependenceMeasure
import csv   
from os.path import exists


random_state = 0


def one_hot(x, num_classes=2):
  return np.squeeze(np.eye(num_classes)[x.reshape(-1)])

def load_data(db_name, train_frac):
    #db_name="bioresponse"
    #db_name="ionosphere"
    #db_name="spambase"
    #db_name="one-hundred-plants-texture" #+
    #db_name="splice" format error
    #db_name="mushroom" format error
    #db_name="lsvt" #+
    #db_name="micro-mass" #+
    #db_name="tokyo1" #+
    #db_name="clean1" #+
    #db_name="tic-tac-toe"


    #eeg-eye-state
    #X,y = fetch_openml(name="ionosphere", as_frame=True, return_X_y=True)
    #X,y = fetch_openml(name="spambase", as_frame=True, return_X_y=True)
    #X,y = fetch_openml(name="one-hundred-plants-texture", as_frame=True, return_X_y=True)
    X,y = fetch_openml(name=db_name, as_frame=True, return_X_y=True)

    print("X shape: {}".format(X.shape))
    if db_name == "ionosphere":
        X = X.to_numpy()[:,2:] # remove constant columns

    #breakpoint()
    dim_x = X.shape[1]
    categories = pd.unique(y.to_numpy().ravel())
    y_num = np.zeros(len(y))
    #print(len(categories))
    #breakpoint()
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
    return X_train, y_train, X_test, y_test, X_train.shape[1], len(categories),X.shape[0]

def benchmark(X_train, y_train, X_test, y_test, method_name = 'R'):

    #knn = KNeighborsClassifier(n_neighbors=3)
    logistic = linear_model.LogisticRegression(max_iter=10000)
    #svc = SVC(kernel='rbf', gamma='auto', degree=2)
    #lsvm = LinearSVC(random_state=0, max_iter=100000)
    result = logistic.fit(X_train, y_train).score(X_test, y_test)
    print("%5.2f" % (result), end=" ")

    #print("LR score(%s): %f" % (method_name, logistic.fit(X_train, y_train).score(X_test, y_test)))
    #print("KNN3 score(%s): %f" % (method_name, knn.fit(X_train, y_train).score(X_test, y_test)))
    #print("LSVM score(%s): %f" % (method_name, lsvm.fit(X_train, y_train).score(X_test, y_test)))
    #print("SVM score(%s): %f" % (method_name, svc.fit(X_train, y_train).score(X_test, y_test)))
    #print("---")
    return result

# seq 10 | xargs -l -- | sed 's/[0-9]\+/pc4/g' | xargs -I {} python ./feature_extraction.py  {}
# seq 25 | xargs -l -- | sed 's/[0-9]\+/robot-failures-lp2/g' | xargs -I {} python ./feature_extraction.py  {}

if __name__ == "__main__":

    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')

    if len(sys.argv) != 2:
        print("Usage {} OpenML_dbname".format(sys.argv[0]))
        sys.exit(0)

    num_epochs = 250 #200
    train_frac = 0.5
    normalize = True


    X_train, y_train, X_test, y_test, dim_x, num_classes, num_samples = load_data(sys.argv[1], train_frac)

    dim_y = num_classes 
    num_features = int(0.5*dim_x) #0.5
    
    n_batch = 1024 #X_train.shape[0]

    kim = KacIndependenceMeasure(dim_x, dim_y, lr=0.007, input_projection_dim = num_features, weight_decay=0.01, orthogonality_enforcer=1.0) #0.007
    #knn = KNeighborsClassifier(n_neighbors=3)
    #logistic = linear_model.LogisticRegression(max_iter=1000)
    #svc = SVC(kernel='poly', gamma='auto', degree=2)
    #lsvm = LinearSVC(random_state=42, max_iter=10000)
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
            print("epoch {} batch {} {}, {}".format(i, j, dep, Xb.shape[0]))

  

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

    #plt.plot(dep_history)
    #plt.show()


    num_features_nca = num_features
    max_num_features_nca = min(X_train.shape[0], X_train.shape[1])
    num_features_nca = min(num_features_nca, max_num_features_nca)
    #breakpoint()

    pca = make_pipeline(StandardScaler(), PCA(n_components=num_features_nca, random_state=random_state))
    pca.fit(X_train, y_train)
    PCA_X_train = pca.transform(X_train)
    PCA_X_test = pca.transform(X_test)

    num_features_nca = num_features
    #max_num_features_nca = min(X_train.shape[0], X_train.shape[1])
    #num_features_nca = min(num_features_nca, max_num_features_nca)
    #breakpoint()

    nca = make_pipeline(StandardScaler(),NeighborhoodComponentsAnalysis(n_components=num_features_nca, random_state=random_state))
    nca.fit(X_train, y_train)
    NCA_X_train = nca.transform(X_train)
    NCA_X_test = nca.transform(X_test)

    #from os.path import exists
    handle_exception = True
    try:
        train = pd.read_csv('feature_extraction/{}.csv'.format(sys.argv[1]))
        train_tensor = torch.tensor(train.values)
        if train_tensor.shape[0] > 27:
            print("Enough!")    
            handle_exception = False         
        else: 
            print("A")
            with open('feature_extraction/{}.csv'.format(sys.argv[1]),'a') as fd:
                print(sys.argv[1],end =" ")
                print(" & ",end =" ")
                print("({},{},{})".format(num_samples,X_train.shape[1],num_classes),end =" ")
                print(" & ",end =" ")
                rez_raw = benchmark(X_train, y_train, X_test, y_test, 'R')
                print(" & ",end =" ")
                rez_kacIMFE = benchmark(F_train, y_train, F_test, y_test, 'KacIMF')
                print(" & ",end =" ")
                #benchmark(PCA_X_train, y_train, PCA_X_test, y_test, 'PCA')
                rez_NCA = benchmark(NCA_X_train, y_train, NCA_X_test, y_test, 'NCA')
                print("\\\\",end ="\n")
                #print("num_classes {}".format(num_classes))

                result_row = [rez_raw, rez_kacIMFE, rez_NCA] 
                writer = csv.writer(fd)
                writer.writerow(result_row)
    except:         
            if handle_exception == False:      
                sys.exit(0)
            else:
                with open('feature_extraction/{}.csv'.format(sys.argv[1]),'a') as fd:
                    print(sys.argv[1],end =" ")
                    print(" & ",end =" ")
                    print("({},{},{})".format(num_samples,X_train.shape[1],num_classes),end =" ")
                    print(" & ",end =" ")
                    rez_raw = benchmark(X_train, y_train, X_test, y_test, 'R')
                    print(" & ",end =" ")
                    rez_kacIMFE = benchmark(F_train, y_train, F_test, y_test, 'KacIMF')
                    print(" & ",end =" ")
                    #benchmark(PCA_X_train, y_train, PCA_X_test, y_test, 'PCA')
                    rez_NCA = benchmark(NCA_X_train, y_train, NCA_X_test, y_test, 'NCA')
                    print("\\\\",end ="\n")
                    #print("num_classes {}".format(num_classes))

                    result_row = [rez_raw, rez_kacIMFE, rez_NCA] 
                    writer = csv.writer(fd)
                    writer.writerow(result_row)  


    """
    print("LR score(R): %f" % logistic.fit(X_train, y_train).score(X_test, y_test))
    print("LR score(F): %f" % logistic.fit(F_train, y_train).score(F_test, y_test))
    print("KNN score(R): %f" % knn.fit(X_train, y_train).score(X_test, y_test))
    print("KNN score(F): %f" % knn.fit(F_train, y_train).score(F_test, y_test))
    print("LSVM score(R): %f" % lsvm.fit(X_train, y_train).score(X_test, y_test))
    print("LSVM score(F): %f" % lsvm.fit(F_train, y_train).score(F_test, y_test))
    print("SVM score(R): %f" % svc.fit(X_train, y_train).score(X_test, y_test))
    print("SVM score(F): %f" % svc.fit(F_train, y_train).score(F_test, y_test))
    """

