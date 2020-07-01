import torch
from torch import nn

from nn.linearsymplectic import LinearSymplectic
from nn.nonlinearsymplectic import LowerNonlinearSymplectic, UpperNonlinearSymplectic, HarmonicUnit

class SympNet(nn.Sequential):
    # TODO make dim consistens as in `models`, phase space has dimension 2, not 1!
    def __init__(self, layers, sub_layers, dim, dt = 0.1):
        modules = []

        # Add upper and lower nonlinear symplectic unit alternately
        # and a linear symplectic unit in-between
        for k in range(layers):
            modules.append(LinearSymplectic(sub_layers, dim, dt))
            if k % 2 == 0:
                modules.append(LowerNonlinearSymplectic(dim, dt, bias=False))
            else:
                modules.append(UpperNonlinearSymplectic(dim, dt, bias=False))
        
        modules.append(LinearSymplectic(sub_layers, dim, dt))
        super(SympNet, self).__init__(*modules)

class HarmonicSympNet(nn.Sequential):
    # TODO make dim consistens as in `models`, phase space has dimension 2, not 1!
    def __init__(self, layers, sub_layers, dim, dt = 0.1):
        modules = []

        # Add upper and lower nonlinear symplectic unit alternately
        # and a linear symplectic unit in-between
        for k in range(layers):
            modules.append(LinearSymplectic(sub_layers, dim, dt))
            modules.append(HarmonicUnit(dt))
        
        modules.append(LinearSymplectic(sub_layers, dim, dt))
        super(SympNet, self).__init__(*modules)

# TODO implement fully connected layer
#class FNN(nn:Sequential):
#    def __init__():
#        pass