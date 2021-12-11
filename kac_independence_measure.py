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

        if self.input_projection_dim > 0: # MNN: 
            self.projection = weight_norm(nn.Linear(self.dim_x, self.input_projection_dim))
            self.dim_x = self.input_projection_dim

        self.reset()

    def reset(self):
        self.a = Variable(torch.randn(self.dim_x), requires_grad=True)
        self.b = Variable(torch.randn(self.dim_y), requires_grad=True)

        if self.input_projection_dim > 0:
            self.optimizer = torch.optim.AdamW(list(self.projection.parameters()) + [self.a, self.b], lr=self.lr, weight_decay=0.0) 
        else:
            self.optimizer = torch.optim.AdamW([self.a, self.b], lr=self.lr, weight_decay=0.0) 

   
    
    def forward(self, x, y):
        # batch norm?
        if self.input_projection_dim > 0:
                x = self.projection(x)


        xa = (x @ self.a/torch.norm(self.a))
        yb = (y @ self.b/torch.norm(self.b))
        f = torch.exp(1j*(xa + yb)).mean() - torch.exp(1j*xa).mean() * torch.exp(1j*yb).mean()
        kim = torch.norm(f)

        loss = -kim # maximise => negative
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()   

        return kim



if __name__ == "__main__":    
    n_batch = 2048 #1024
    dim_x = 1024
    dim_y = 32
    num_iter = 500 #1300
    input_proj_dim = 0

    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    
    """
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
    

    # dependent data
    print("Dependent data")
    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y)
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        #y = (torch.sin(proj_x) + torch.cos(proj_x))*noise    
        #y = torch.log(1.0 + torch.abs(proj_x))
        y = torch.sin(proj_x) + torch.cos(proj_x)  + 0.2*noise
        #y = x
        dep = model(x,y)
        history_dep.append(dep.detach().numpy())
        #print("{} {}".format(i, dep))
    plt.plot(history_dep, label="Dependent")
    plt.savefig('./additive_noise_effect_0.1.png')
    """

    file_object = open('noise_effect.txt', 'w')

    to_plot = []
    step = 0
    for noise_level in np.arange(0, 2.5, 0.1):
        print("Noise level: {}".format(noise_level))
        history_dep = []
        random_proj = nn.Linear(dim_x, dim_y)
        model.reset()
        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y)
            #y = (torch.sin(proj_x) + torch.cos(proj_x))*noise    
            #y = torch.log(1.0 + torch.abs(proj_x))
            y = torch.sin(proj_x) + torch.cos(proj_x)  + noise_level*noise
            #y = x
            dep = model(x,y)
            history_dep.append(dep.detach().numpy())
            #print("{} {}".format(i, dep))
        to_plot.append(history_dep[-1])
        file_object.write(str(history_dep[-1]))    
        file_object.write('\n')
        plt.plot(history_dep, label="Dependent")
        plt.savefig('./additive_noise_effect_{}.png'.format(step))
        step = step + 1
        plt.clf()

    breakpoint()
    plt.plot(np.arange(0, 2.5, 0.1), to_plot)    
    plt.savefig('./additive_noise_effects.png')

    file_object.close()












