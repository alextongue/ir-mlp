import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import datetime
# import pysofaconventions as sofa
import pdb
import scipy.io as sio
from utils.irmlp_core import IRMLP
from utils.hrir_math import *

class Dataset_HRIR(Dataset):
    def __init__(self, path='', fourier_depth=10, batch_size=128):

        self.path=path
        self.fourier_depth=fourier_depth
        self.batch_size=batch_size

        # self.sofa = sofa.SOFAFile(self._path, 'r')
        infile = sio.loadmat(self.path, )

        self.ir = infile['ir']
        self.fs = int(infile['fs'])
        self.srcpos = infile['srcpos']

        self.num_irs = self.ir.shape[0]
        self.irlen = self.ir.shape[2]
        self.nn = np.arange(self.irlen)

        self.fourier_ir_L = np.squeeze(self.ir[:,0,:])
        self.fourier_ir_R = np.squeeze(self.ir[:,1,:])
        self.fourier_nn = fourier_features(np.linspace(-1,1,self.irlen), \
                                           L=self.fourier_depth)

        self.srcpos_cart = sph2cart(self.srcpos, deg2rad=True)
        self.fourier_xx = fourier_features(self.srcpos_cart[:,0], L=self.fourier_depth)
        self.fourier_yy = fourier_features(self.srcpos_cart[:,1], L=self.fourier_depth)
        self.fourier_zz = fourier_features(self.srcpos_cart[:,2], L=self.fourier_depth)

        self._input = []
        self._output = []

        #pdb.set_trace()

        for ii in range(self.fourier_xx.shape[0]):
            for nn in range(self.fourier_nn.shape[0]):
                self._input.append(
                    np.concatenate(( \
                        self.fourier_nn[nn], 
                        self.fourier_xx[ii],
                        self.fourier_yy[ii],
                        self.fourier_zz[ii]),
                        axis=0)
                )
                self._output.append(self.ir[ii,:,nn])

        pdb.set_trace()
        
    def __len__(self):
        return len(self._input)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        return torch.tensor(self._input[idx]), torch.tensor(self._output[idx])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="IR MLP")

    parser.add_argument('-c', '--cuda-id', default=None, type=int)
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    parser.add_argument('-fd', '--fourier-depth', default=10, type=int)
    parser.add_argument('-ln', '--layer-num', default=6, type=int)
    parser.add_argument('-ld', '--layer-depth', default=128, type=int)
    parser.add_argument('-lr', '--learning-rate', default=5e-5, type=float)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-pt', '--training-path', default='./utils/KU100_HRTF_simple.mat', type=str)

    savestr = datetime.date.today().strftime("%Y%m%d")
    parser.add_argument('-ps', '--save-path', default='save/'+savestr+'.save')

    args = parser.parse_args()

    if args.cuda_id == None:
        device_str = 'cpu'
    else:
        device_str = 'cuda:'+cuda_id

    # Initialize Dataloaders
    data_train = Dataset_HRIR(args.training_path, args.fourier_depth, args.batch_size)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, num_workers=0)

    pdb.set_trace()

    # Initialize NN
    irmlp = IRMLP(input_size=128, output_size=128, \
        mlp_depth=args.layer_num, \
        fourier_depth=args.fourier_depth, cuda_id=args.cuda_id).to(device_str)

    # Train NN
    loss = nn.MSELoss()
    optimizer = optim.Adam(irmlp.parameters(), lr=args.learning_rate)

    irmlp.train() # set to training mode