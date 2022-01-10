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

from kac_independence_measure import KacIndependenceMeasure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn import datasets, neighbors, linear_model

if __name__ == "__main__":

    if not os.path.exists('feature_extraction'):
        os.makedirs('feature_extraction')

    n_batch = 256 #2048
    dim_x = 64 # 1024
    dim_y = 1 # 32
    num_iter = 500 #500
    input_proj_dim = 0 #0

    X_digits, y_digits = datasets.load_digits(return_X_y=True)
    X_digits = X_digits / X_digits.max()

    n_samples = len(X_digits)

    X_train = X_digits[: int(0.9 * n_samples)]
    y_train = y_digits[: int(0.9 * n_samples)]
    X_test = X_digits[int(0.9 * n_samples) :]
    y_test = y_digits[int(0.9 * n_samples) :]

    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    knn = KNeighborsClassifier(n_neighbors=3)
    logistic = linear_model.LogisticRegression(max_iter=1000)
    #knn.fit(X_train, y_train)

    print("Raw data")
    print("KNN score: %f" % knn.fit(X_train, y_train).score(X_test, y_test))
    print(
        "LogisticRegression score: %f"
        % logistic.fit(X_train, y_train).score(X_test, y_test)
    )
    breakpoint()

    num_iter = 1024

    #for i in range(num_iter):
    #    model.forward(Xb, Yb)

    