import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import pdb

class IRMLP(nn.Module):
    def __init__(self, output_size=128, input_size=128, batch_size=256, mlp_depth=6, cuda_id=False):
        
        super(IRMLP, self).__init__()

        use_cuda = False
        if cuda_id != False:
            use_cuda = True
        
        self.network = nn.ModuleList()

        for i in range(mlp_depth):
            self.network.append(nn.Linear(output_size,output_size))
            self.network.append(nn.LeakyReLU())

        self.network.append(nn.Linear(output_size,1))

    # pdb.set_trace()

    def forward(self,x):
        for layer in self.network:
            x = layer(x)
        return x