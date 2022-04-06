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


from kac_independence_measure import KacIndependenceMeasure

device = "cuda:0"


if __name__ == "__main__":

    if not os.path.exists('basic_demonstration'):
        os.makedirs('basic_demonstration')

    n_batch = 8192 #8192
    #dim_x = 512 # 1024
    dim_y = 512 # 32
    num_iter = 250 #200
    input_proj_dim = 0 #dim_x #0
    lr = 0.05 #0.05

    final_values_indep = []
    final_values_additive = []
    final_values_multiplicative = []
    xx = []

    for dim_x in range(2,512,10):
        print("dim_x = {}".format(dim_x))
        
        fig = plt.figure()
        plt.figure().clear()

        xx.append(dim_x)

        #model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device=device)

        
    
        # inedependent data
        history_indep = []
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            y = torch.randn(n_batch, dim_y)
            dep = model(x,y)
            history_indep.append(dep.cpu().detach().numpy())
            #print("{} {}".format(i, dep))       
        #plt.plot(history_indep, label='Independent')

        final_values_indep.append(history_indep[-1])
        plt.plot(xx, final_values_indep, label="Independent")

        """
        x = torch.randn(n_batch, dim_x)
        y = torch.randn(n_batch, dim_y)    
        dep_dcor = dcor.distance_covariance(x, y)
        dep_dcor_u = dcor.u_distance_covariance_sqr(x, y)
        print("Independent: Dcor(x,y) = {}, Dcor_u(x,y) = {}".format(dep_dcor, dep_dcor_u))
        """

        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr,  input_projection_dim = input_proj_dim, weight_decay=0.01, device=device)
        

        # dependent data (additive noise)
        history_dep = []
        random_proj = nn.Linear(dim_x, dim_y)
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y)
            y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise    
            dep = model(x,y)
            history_dep.append(dep.cpu().detach().numpy())
            #print("{} {}".format(i, dep))
        
        """"
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise        
        x = x.detach().numpy()
        y = y.detach().numpy()
        dep_dcor = dcor.distance_covariance(x, y)
        dep_dcor_u = dcor.u_distance_covariance_sqr(x, y)    
        print("Additive noise: Dcor(x,y) = {}, Dcor_u(x,y) = {}".format(dep_dcor, dep_dcor_u))
        """

        final_values_additive.append(history_dep[-1])

        #plt.plot(history_dep, label="Dependent_additive")
        
        plt.plot(xx, final_values_additive, label="Dependent_additive")

        """
        model = KacIndependenceMeasure(dim_x, dim_y, lr=lr, input_projection_dim = input_proj_dim, weight_decay=0.01, device=device)


        history_dep = []
        random_proj = nn.Linear(dim_x, dim_y)
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y)
            y = (torch.sin(proj_x) + torch.cos(proj_x))*noise    
            dep = model(x,y)
            history_dep.append(dep.cpu().detach().numpy())
            #print("{} {}".format(i, dep))

        

        #plt.plot(history_dep, label="Dependent_multiplicative")

        final_values_multiplicative.append(history_dep[-1])

        plt.plot(xx, final_values_multiplicative, label="Dependent_multiplicative")
        """

        plt.savefig('./basic_demonstration/aaa_dependence_detection_kacim_by_dim_{}.png'.format(dim_x))
        plt.close(fig)

