import numpy as np
import pdb

def sph2cart(coords_sph, deg2rad=False):

    azi = coords_sph[:,0]
    ele = coords_sph[:,1]
    r   = coords_sph[:,2]

    if deg2rad:
        azi = azi*np.pi/180
        ele = ele*np.pi/180

    x = np.multiply( np.multiply(r, np.cos(ele)), np.cos(azi) )
    y = np.multiply( np.multiply(r, np.cos(ele)), np.sin(azi) )
    z = np.multiply(r, np.sin(ele))

    return np.stack([x,y,z]).T

def fourier_features(xx, L=10):

    xx_scaled = 2 * (xx-xx.min())/(xx.max()-xx.min()) - 1

    base2 = np.logspace(0,L-1,L,base=2)

    ff_sin = np.sin(np.pi * xx_scaled[:,np.newaxis] * base2)
    ff_cos = np.cos(np.pi * xx_scaled[:,np.newaxis] * base2)

    ff = np.zeros((xx.shape[0],L*2))
    ff[:,::2] = ff_sin
    ff[:,1::2] = ff_cos

    return ff