import torch
from torch import nn

from nn.linearsymplectic import LinearSymplectic
from nn.nonlinearsymplectic import LowerNonlinearSymplectic, UpperNonlinearSymplectic, HarmonicUnit

from models.vectors import NumpyPhaseSpace, PhaseSpaceVectorList

class StepIntegrator:

    # TODO hard-coded for 2 dimensions right now...
    def integrate(self, q0, p0, t_start, t_end, dt, device=None):
        phase_space = NumpyPhaseSpace(2)
        result = PhaseSpaceVectorList()

        t_curr = t_start
        q_curr = q0
        p_curr = p0

        result.append(t_curr, phase_space.new_vector(q_curr, p_curr))
        
        # replicate behavior of original models 
        # and thus interpret t_end as an open interval
        while t_curr <= t_end - dt:
            y = self.__call__(torch.tensor([[q_curr, p_curr]]).to(device))

            t_curr += dt
            q_curr = y.data[0][0].item()
            p_curr = y.data[0][1].item()

            result.append(t_curr, phase_space.new_vector(q_curr, p_curr))  

        return result

class SympNet(nn.Sequential, StepIntegrator):
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

class LinearSympNet(nn.Sequential, StepIntegrator):
    # TODO make dim consistens as in `models`, phase space has dimension 2, not 1!
    def __init__(self, layers, sub_layers, dim, dt = 0.1):
        modules = []

        for k in range(layers):
            modules.append(LinearSymplectic(sub_layers, dim, dt))
        
        super(LinearSympNet, self).__init__(*modules)

class HarmonicSympNet(nn.Sequential, StepIntegrator):
    # TODO make dim consistens as in `models`, phase space has dimension 2, not 1!
    def __init__(self, layers, sub_layers, dim, dt = 0.1):
        modules = []

        for k in range(layers):
            modules.append(LinearSymplectic(sub_layers, dim, dt))
            modules.append(HarmonicUnit(dt))
        
        modules.append(LinearSymplectic(sub_layers, dim, dt))
        super(HarmonicSympNet, self).__init__(*modules)

# TODO implement fully connected layer
#class FNN(nn:Sequential):
#    def __init__():
#        pass