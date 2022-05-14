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


    num_runs = 25
    n_batch = 8192 
    dim_x = 512 
    dim_y = 512
    num_iter = 200 
    input_proj_dim = 0 
    lr = 0.05


    model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01)
    
   
    # inedependent data
    history_indep = []
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        y = torch.randn(n_batch, dim_y)
        dep = model(x,y)
        history_indep.append(dep.detach().numpy())
    plt.plot(history_indep)  

    
    model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01)
    

    # data with statistical dependence
    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y)
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise    
        dep = model(x,y)
        history_dep.append(dep.detach().numpy())
    
    plt.plot(history_dep) 
    



    plt.savefig('./basic_demonstration/dependence_detection_{}_{}.png'.format(dim_x, dim_y))

