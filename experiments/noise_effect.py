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

from kac_independence_measure import KacIndependenceMeasure
import os


if __name__ == "__main__":

    if not os.path.exists('noise_effect'):
        os.makedirs('noise_effect')

    n_batch = 2048 #2048
    dim_x = 512 # 1024
    dim_y = 4 # 32
    num_iter = 1000 #500
    input_proj_dim = 0#64 #0

    #model = KacIndependenceMeasure(dim_x, dim_y, lr=0.05, num_iter = num_iter, input_projection_dim = input_proj_dim)
    #model = KacIndependenceMeasure(dim_x, dim_y, lr=0.01, num_iter = num_iter, input_projection_dim = input_proj_dim)

    file_object = open('./noise_effect/noise_effect.txt', 'w')

    to_plot = []
    step = 0

    random_proj = nn.Linear(dim_x, dim_y) #, requires_grad=False)

    noise_levels = np.arange(0.0, 3.0, 0.1)

    for noise_level in noise_levels:
        print("noise_level: {}".format(noise_level))
        history_dep = []
        #model.reset()

        model = KacIndependenceMeasure(dim_x, dim_y, lr=0.01,  input_projection_dim = input_proj_dim, weight_decay=0.001, device="cuda:0")

        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y)
            y = torch.sin(proj_x) + torch.cos(proj_x)  + noise_level*noise           
            dep = model(x.to("cuda:0"),y.to("cuda:0"))
            history_dep.append(dep.cpu().detach().numpy())
            #print("{} {}".format(i, dep))
        to_plot.append(history_dep[-1])
        file_object.write(str(history_dep[-1]))    
        file_object.write('\n')
        plt.plot(history_dep, label="Scale: ".format(str(noise_level)))
        plt.savefig('./noise_effect/{}.png'.format(step))
        step = step + 1
        plt.clf()

    #breakpoint()
    plt.plot(noise_levels, to_plot)    
    plt.savefig('./noise_effect/summary.png')

    file_object.close()