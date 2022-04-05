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
import dcor
from scipy.stats import wilcoxon

from kac_independence_measure import KacIndependenceMeasure


if __name__ == "__main__":

    if not os.path.exists('basic_demonstration'):
        os.makedirs('basic_demonstration')

    n_batch = 2048 #2048
    dim_x = 512 # 1024
    dim_y = 512 # 32
    num_iter = 50 #500
    input_proj_dim = 0 #dim_x #0
    lr = 0.05

    #model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device="cuda:0")

    
   
    # inedependent data
    history_indep = []
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x).detach().numpy()
        y = torch.randn(n_batch, dim_y).detach().numpy()
        #dep = model(x,y)
        dep_dcor = dcor.distance_covariance(x, y)
        #dep_dcor_u = dcor.u_distance_covariance_sqr(x, y)
        #history_indep.append(dep.detach().numpy())
        history_indep.append(dep_dcor)
        #print("{} {}".format(i, dep))       
        print(i)
    plt.plot(history_indep, label='Independent')
    

    

    model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01)
    

    # dependent data (additive noise)
    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y)
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise    
        #dep = model(x,y)
        dep_dcor = dcor.distance_covariance(x.detach().numpy(), y.detach().numpy())
        #dep_dcor_u = dcor.u_distance_covariance_sqr(x, y)
        history_dep.append(dep_dcor)
        #print("{} {}".format(i, dep))
        print(i)
    

    
    plt.plot(history_dep, label="Dependent_additive")


    plt.savefig('./basic_demonstration/dependence_detection_dcor.png')


    w, p = wilcoxon(history_dep, history_indep, alternative="greater")

    x = torch.randn(n_batch, dim_x)
    proj_x = random_proj(x)
    noise = torch.randn(n_batch, dim_y)
    y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise    
    z = torch.randn(n_batch, dim_y).detach().numpy()
    x = x.detach().numpy()
    y = y.detach().numpy()
    

    breakpoint()
