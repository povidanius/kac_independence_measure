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

from kac_independence_measure import KacIndependenceMeasure


if __name__ == "__main__":

    if not os.path.exists('basic_demonstration'):
        os.makedirs('basic_demonstration')

    n_batch = 2048 #2048
    dim_x = 512 # 1024
    dim_y = 4 # 32
    num_iter = 500 #500
    input_proj_dim = 0 #0

    #model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)

    
   
    # inedependent data
    history_indep = []
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        y = torch.randn(n_batch, dim_y)
        dep = model(x,y)
        history_indep.append(dep.detach().numpy())
        #print("{} {}".format(i, dep))
    plt.plot(history_indep, label='Independent')

    

    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    

    # dependent data (additive noise)
    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y)
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        #y = (torch.sin(proj_x) + torch.cos(proj_x))*noise    
        #y = torch.log(1.0 + torch.abs(proj_x))
        y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise
        #y = x
        dep = model(x,y)
        history_dep.append(dep.detach().numpy())
        #print("{} {}".format(i, dep))
    plt.plot(history_dep, label="Dependent_additive")

    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)


    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y)
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        y = (torch.sin(proj_x) + torch.cos(proj_x))*noise    
        #y = torch.log(1.0 + torch.abs(proj_x))
        #y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise
        #y = x
        dep = model(x,y)
        history_dep.append(dep.detach().numpy())
        #print("{} {}".format(i, dep))
    plt.plot(history_dep, label="Dependent_multiplicative")


    plt.savefig('./basic_demonstration/dependence_detection.png')

