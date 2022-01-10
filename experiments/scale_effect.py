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

from kac_independence_measure import KacIndependenceMeasure
import os


if __name__ == "__main__":

    if not os.path.exists('scale_effect'):
        os.makedirs('scale_effect')

    n_batch = 2048 #2048
    dim_x = 512 # 1024
    dim_y = 4 # 32
    num_iter = 1000 #500
    input_proj_dim = 0#64 #0

    input_normalization = True


    file_object = open('./scale_effect/scale_effect.txt', 'w')

    to_plot = []
    step = 0

    scales = np.arange(0.1, 2., 0.1)
    random_proj = nn.Linear(dim_x, dim_y)
    for scale in scales:
        print("scale_level: {}".format(scale))
        history_dep = []
    
        model = KacIndependenceMeasure(dim_x, dim_y, lr=0.01, num_iter = num_iter, input_projection_dim = input_proj_dim)
        #mf = scipy.signal.medfilt()

        for i in range(num_iter):
            x = torch.randn(n_batch, dim_x)
            proj_x = random_proj(x)
            noise = torch.randn(n_batch, dim_y)
            #y = (torch.sin(proj_x) + torch.cos(proj_x))*noise    
            #y = torch.log(1.0 + torch.abs(proj_x))
            y = torch.sin(proj_x) + torch.cos(proj_x)  + 1.0*noise           
            #y = noise*x #+ noise_level*noise
            dep = model(scale*x,scale*y, input_normalization)
            history_dep.append(dep.detach().numpy())
            #print("{} {}".format(i, dep))
        history_dep = mf(history_dep, 17)            
        to_plot.append(history_dep[-1])
        file_object.write(str(history_dep[-1]))    
        file_object.write('\n')
        plt.plot(history_dep, label="Scale: ".format(str(scale)))
        plt.savefig('./scale_effect/{}.png'.format(step))
        step = step + 1
        plt.clf()

    #breakpoint()    
    plt.plot(scales, to_plot)    
    plt.savefig('./scale_effect/summary.png')

    file_object.close()