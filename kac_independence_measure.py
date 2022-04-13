import torch
import torch.nn as nn
import torch.nn.functional as F
#import math
#import numpy as np
#import matplotlib.pyplot as plt
from torch.autograd import Variable
import itertools
#from torch.nn.utils import weight_norm
#import numpy as np

# Kac independence measure, preprint/article will be added further.
# Author: Dr. Povilas Daniušis, ORCID 0000-0001-5977-827X
# Special thanks to Neurotechnology (www.neurotechnology.com)


class KacIndependenceMeasure(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005,  input_projection_dim = 0, output_projection_dim=0, weight_decay=0.01, orthogonality_enforcer = 1.0, device="cpu"):
        super(KacIndependenceMeasure, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim
        self.output_projection_dim = output_projection_dim
        self.weight_decay = weight_decay
        self.orthogonality_enforcer = orthogonality_enforcer
        self.device = device
        self.reset()

    def reset(self):


        param_list = []
        if self.input_projection_dim > 0:
            self.a = Variable(torch.rand(self.input_projection_dim,device=self.device), requires_grad=True)
            #self.projection = weight_norm(nn.Linear(self.dim_x, self.input_projection_dim))
            self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_x.parameters()) #  + [self.a] #[self.a, self.b]
        else:
            self.a = Variable(torch.ones(self.dim_x, device=self.device), requires_grad=True)

        if self.output_projection_dim > 0:
            self.b = Variable(torch.rand(self.output_projection_dim,device=self.device), requires_grad=True)
            self.projection_y = nn.Linear(self.dim_y, self.output_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_y.parameters()) #+ [self.b]
        else:            
            self.b = Variable(-torch.ones(self.dim_y,device=self.device), requires_grad=True)
        
        self.trainable_parameters = param_list  + [self.a, self.b]        
        self.optimizer = torch.optim.AdamW(param_list  + [self.a, self.b], lr=self.lr, weight_decay=self.weight_decay) 


   

    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = (x - x.mean(axis=0, keepdim=True))/(0.00001 + x.std(axis=0, keepdim=True))

        proj = self.projection_x(x)            
        return proj

    def forward(self, x, y, update = True, normalize=True):
        x = x.to(self.device)
        y = y.to(self.device)
        if normalize:
            x = (x - x.mean(axis=0, keepdim=True))/(0.00001 + x.std(axis=0, keepdim=True))
            #if self.dim_y > 1:
            #    y = (y - y.mean(axis=1, keepdim=True))/y.std(axis=1, keepdim=True)
            #else:    
            y = (y - y.mean(axis=0, keepdim=True))/(0.00001 + y.std(axis=0, keepdim=True))
        

        if self.input_projection_dim > 0:                
                x = self.projection_x(x)
        if self.output_projection_dim > 0:
                y = self.projection_y(y)

        xa = (x @ (self.a/torch.norm(self.a)))
        yb = (y @ (self.b/torch.norm(self.b)))

        f = torch.exp(1j*(xa + yb)).mean() - torch.exp(1j*xa).mean() * torch.exp(1j*yb).mean()
        kim = torch.norm(f)

        if update:
            loss = -kim 
            if self.input_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_x.weight,self.projection_x.weight.T) - torch.eye(self.input_projection_dim).to(self.device)) # maximise => negative
            
            if self.output_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_y.weight,self.projection_y.weight.T) - torch.eye(self.output_projection_dim).to(self.device)) # maximise => negative

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

        return kim


if __name__ == "__main__":    
    print("See ./experiments")