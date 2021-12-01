import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Kac independence measure, preprint/article will be added further.
# Author: Dr. Povilas Daniušis, ORCID 0000-0001-5977-827X
# Special thanks to Neurotechnology (www.neurotechnology.com)


class KacIndependenceMeasure(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005, num_iter = 100):
        super(KacIndependenceMeasure, self).__init__()
        self.num_iter = num_iter
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.reset()

    def reset(self):
        self.a = Variable(torch.randn(self.dim_x), requires_grad=True)
        self.b = Variable(torch.randn(self.dim_y), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.a, self.b], lr=self.lr)


    def forward(self, x, y):
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
    n_batch = 256
    dim_x = 1024
    dim_y = 4
    num_iter = 1000

    model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter)
    
    # inedependent data
    print("Independent data")
    history_indep = []
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        y = torch.randn(n_batch, dim_y)
        dep = model(x,y)
        history_indep.append(dep.detach().numpy())
        print("{} {}".format(i, dep))
    plt.plot(history_indep)
    plt.show()
    

    model.reset()    
    
    # dependent data
    print("Dependent data")
    history_dep = []
    random_proj = nn.Linear(dim_x, dim_y)
    for i in range(num_iter):
        x = torch.randn(n_batch, dim_x)
        proj_x = random_proj(x)
        noise = torch.randn(n_batch, dim_y)
        y = torch.sin(proj_x) + torch.cos(proj_x) + noise
        dep = model(x,y)
        history_dep.append(dep.detach().numpy())
        print("{} {}".format(i, dep))
    plt.plot(history_dep)        
    plt.show()














