import argparse
import torch
import torch.nn as nn
import numpy as np
import time


class IRMLP(nn.Module):
    def __init__(self, output_size=256, input_size=259, batch_size=256, mlp_depth=32, cuda_id=False):

        use_cuda = False
        if cuda_id != False:
            use_cuda = True
        
        self.network = nn.ModuleList()

        for i in range(mlp_depth):
            self.network.append(nn.Linear)