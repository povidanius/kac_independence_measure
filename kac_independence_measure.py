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

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Kac independence measure, preprint/article will be added further.
# Author: Dr. Povilas Daniušis, ORCID 0000-0001-5977-827X
# Special thanks to Neurotechnology (www.neurotechnology.com)


class KacIndependenceMeasure(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005, num_iter = 100, input_projection_dim = 0):
        super(KacIndependenceMeasure, self).__init__()
        self.num_iter = num_iter
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim

        self.reset()

    def reset(self):
        self.b = Variable(torch.randn(self.dim_y), requires_grad=True)

        if self.input_projection_dim > 0:
            self.a = Variable(torch.randn(self.input_projection_dim), requires_grad=True)
            self.projection = weight_norm(nn.Linear(self.dim_x, self.input_projection_dim))
            self.optimizer = torch.optim.AdamW(list(self.projection.parameters()) + [self.a, self.b], lr=self.lr, weight_decay=0.1) 
        else:
            self.a = Variable(torch.randn(self.dim_x), requires_grad=True)
            self.optimizer = torch.optim.AdamW([self.a, self.b], lr=self.lr, weight_decay=0.1) 

   



    def forward(self, x, y, update = True, normalize=True):
        if normalize:
            x = (x - x.mean(axis=1, keepdim=True))/x.std(axis=1, keepdim=True)
            y = (y - y.mean(axis=1, keepdim=True))/y.std(axis=1, keepdim=True)
   
        # batch norm?
        if self.input_projection_dim > 0:
                x = self.projection(x)


        xa = (x @ (self.a/torch.norm(self.a)))
        yb = (y @ (self.b/torch.norm(self.b)))
        f = torch.exp(1j*(xa + yb)).mean() - torch.exp(1j*xa).mean() * torch.exp(1j*yb).mean()
        kim = torch.norm(f)

        if update:
            loss = -kim # maximise => negative
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

        return kim



if __name__ == "__main__":    
    n_batch = 2048 #2048
    dim_x = 512 # 1024
    dim_y = 4 # 32
    num_iter = 500 #500
    input_proj_dim = 0 #0

    #model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)

    
   
    # inedependent data
    print("Independent data")
    history_indep = []
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        y = torch.randn(n_batch, dim_y)
        dep = model(x,y)
        history_indep.append(dep.detach().numpy())
        #print("{} {}".format(i, dep))
    plt.plot(history_indep, label='Independent')

    

    model.reset()    
    

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

    model.reset()

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


    plt.savefig('./independent_dependent.png')

