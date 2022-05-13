import sys
sys.path.insert(0, "../")
import math
import numpy as np
import pandas as pd

import torch

from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


from kac_independence_measure import KacIndependenceMeasure

np.random.seed(0)
torch.manual_seed(0)
random_state = 0

# TODO:
# Fix randomness issue DONE
# Fix issue povilas pointed out DONE
# Make test set DONE
# Run on test set DONE
# Fix number of epochs DONE
# Make variable names clear
# Print results for all dbs DONE
# Stop printing epoch number DONE
# Extend to databases DONE
# Remove break DONE

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
    #db_name="tic-tac-toe" #could not convert string to float: 'b'


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=None, shuffle=True, train_size=train_frac) #int(train_frac*X.shape[0]))
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    return X_train, y_train, X_test, y_test, X_train.shape[1], len(categories),X.shape[0]

def load_only(db_name):
    X,y = fetch_openml(name=db_name, as_frame=True, return_X_y=True)
    # Preprocess little
    if db_name == "ionosphere":
        X = X.to_numpy()[:,2:] # remove constant columns
    dim_x = X.shape[1]
    categories = pd.unique(y.to_numpy().ravel())
    y_num = np.zeros(len(y))
    category_id = 0
    for cat in categories:
        ind = np.where(y == cat)
        for ii in ind:
            y_num[ii] = category_id
        category_id = category_id + 1
    y = y_num.astype(np.int32)
    dim_y = len(categories)

    X_train, X_test, y_train, y_test = train_test_split(X, y, \
        stratify=y, random_state=random_state, shuffle=True, train_size=0.80)
    return X_train, y_train, X_test, y_test, dim_x, dim_y

def one_hot(x, num_classes=2):
  return np.squeeze(np.eye(num_classes)[x.reshape(-1)])


def kac_features(X_train, y_train, X_test, y_test, dim_x, dim_y, feature_frac):
    num_epochs = 200
    num_features = int(feature_frac*dim_x) #0.5
    n_batch = 1024
    normalize = True

    kim = KacIndependenceMeasure(dim_x, dim_y, lr=0.007, input_projection_dim=num_features, weight_decay=0.01, orthogonality_enforcer=0.0) #0.007

    # One hot encode
    n = X_train.shape[0]
    ytr = one_hot(y_train, dim_y)
    yte = one_hot(y_test, dim_y)

    # [[ Train kac model ]]
    dep_history = []
    for i in range(num_epochs):
        # if i % 10 == 0:
        #     print("Epoch: {}".format(i))
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
            # print("epoch {} batch {} {}, {}".format(i, j, dep, Xb.shape[0]))

    # Projecting features
    F_train = kim.project(torch.from_numpy(X_train.astype(np.float32)), normalize=normalize).detach().numpy()
    F_test = kim.project(torch.from_numpy(X_test.astype(np.float32)), normalize=normalize).detach().numpy() #test_data.tensors[0]).detach().numpy()

    return F_train, F_test, kim

def pca_features(X_train, y_train, X_test, y_test, dim_x, dim_y, feature_frac):
    num_features = int(feature_frac*dim_x) #0.5

    num_features_nca = num_features
    max_num_features = min(X_train.shape[0], X_train.shape[1])
    num_features = min(num_features, max_num_features)

    pca = make_pipeline(StandardScaler(), PCA(n_components=num_features, random_state=random_state))
    pca.fit(X_train, y_train)
    PCA_X_train = pca.transform(X_train)
    PCA_X_test = pca.transform(X_test)

    return PCA_X_train, PCA_X_test, pca

def nca_features(X_train, y_train, X_test, y_test, dim_x, dim_y, feature_frac):
    # Neighborhood Component Analysis
    num_features = int(feature_frac*dim_x)
    
    nca = make_pipeline(StandardScaler(), \
        NeighborhoodComponentsAnalysis(n_components=num_features, random_state=random_state))
    nca.fit(X_train, y_train)
    NCA_X_train = nca.transform(X_train)
    NCA_X_test = nca.transform(X_test)

    return NCA_X_train, NCA_X_test, nca

def benchmark(X_train, y_train, X_test, y_test): 
    logistic = linear_model.LogisticRegression(max_iter=10000)
    result = logistic.fit(X_train, y_train).score(X_test, y_test)
    # print("%5.2f" % (result), end=" ")
    # print(result)
    return result

dbs = ['bioresponse','ionosphere','spambase','one-hundred-plants-texture','lsvt','micro-mass', \
    'tokyo1','clean1']
    
for db in dbs:
    print(f"Dataset: {db}")
    for feature_frac in np.arange(0.1,1.1,0.1):#[0.25, 1]:#0.5, 0.75, 1]:
        splits = KFold(4, True, 0)
        # Load dataset
        X, y, X_test, y_test, dim_x, dim_y = load_only(db) # TODO: Make testset

        kac_scores, pca_scores, nca_scores = [], [], []
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(X)))):
            # Index data
            train_x, train_y, valid_x, valid_y = X.iloc[train_idx], y[train_idx], X.iloc[val_idx], y[val_idx]
            # Preprocess
            train_x, valid_x = preprocessing.normalize(train_x), preprocessing.normalize(valid_x)
            # kac
            F_train, F_valid, kac_model = kac_features(train_x, train_y, valid_x, valid_y, dim_x, dim_y, feature_frac)
            res = benchmark(F_train, train_y, F_valid, valid_y)
            # print(f"KAC score = {res}")
            kac_scores.append(res)
            # PCA
            F_train, F_valid, pca_model = pca_features(train_x, train_y, valid_x, valid_y, dim_x, dim_y, feature_frac)
            res = benchmark(F_train, train_y, F_valid, valid_y)
            # print(f"PCA score = {res}")
            pca_scores.append(res)
            # NCA
            F_train, F_valid, nca_model = nca_features(train_x, train_y, valid_x, valid_y, dim_x, dim_y, feature_frac)
            res = benchmark(F_train, train_y, F_valid, valid_y)
            # print(f"NCA score = {res}")
            nca_scores.append(res)
            # break
        
        print(f"Feature size: {feature_frac} kac score: {np.mean(kac_scores):2f} pca score: {np.mean(pca_scores):2f} nca score: {np.mean(nca_scores):2f}")

        # Run test!
        X, X_test  = preprocessing.normalize(X), preprocessing.normalize(X_test)

        # KAC
        F_train, F_test, kac_model = kac_features(X, y, X_test, y_test, dim_x, dim_y, feature_frac)
        kac_res = benchmark(F_train, y, F_test, y_test)

        # PCA
        F_train, F_test, kac_model = pca_features(X, y, X_test, y_test, dim_x, dim_y, feature_frac)
        pca_res = benchmark(F_train, y, F_test, y_test)

        # NCA
        F_train, F_test, kac_model = nca_features(X, y, X_test, y_test, dim_x, dim_y, feature_frac)
        nca_res = benchmark(F_train, y, F_test, y_test)

        print(f"Test score: kac {kac_res:2f} pca {pca_res:2f} nca {nca_res:2f}")
