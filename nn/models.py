import torch
from torch import nn
from collections import OrderedDict

from nn.linearsymplectic import LinearSymplectic, LowerSymplecticConv1d, UpperSymplecticConv1d
from nn.nonlinearsymplectic import LowerNonlinearSymplectic, UpperNonlinearSymplectic, HarmonicUnit

from models.vectors import NumpyPhaseSpace, PhaseSpaceVectorList

class StepIntegrator:

    def integrate(self, q0, p0, t_start, t_end, dt, device=None, custom_phase_space=None):
        assert self.dim, 'self.dim must be set in __init__()'

        dim_half = int(self.dim/2)
        phase_space = NumpyPhaseSpace(self.dim) if custom_phase_space is None else custom_phase_space
        result = PhaseSpaceVectorList()

        t_curr = t_start
        x_curr = torch.cat([
            torch.tensor(q0, dtype=torch.float32).reshape(1,dim_half), 
            torch.tensor(p0, dtype=torch.float32).reshape(1,dim_half), 
            ], dim=1).to(device)

        result.append(t_curr, phase_space.new_vector(q0, p0))
        
        # replicate behavior of original models 
        # and thus interpret t_end as an open interval
        while t_curr <= t_end - dt:
            x_curr = self.__call__(x_curr)
            t_curr += dt

            # TODO why detach() required if original x_curr does not require grad?
            x_curr_numpy = x_curr.detach().numpy()
            q_curr = x_curr_numpy[0, 0:dim_half]
            p_curr = x_curr_numpy[0, dim_half:self.dim]
            result.append(t_curr, phase_space.new_vector(q_curr, p_curr))  

        return result

class SympNet(nn.Sequential, StepIntegrator):

    def __init__(self, layers, sub_layers, dim, activation_fn=torch.sigmoid):
        self.dim = dim
        modules = []

        # Add upper and lower nonlinear symplectic unit alternately
        # and a linear symplectic unit in-between
        for k in range(layers):
            modules.append(LinearSymplectic(sub_layers, dim))
            if k % 2 == 0:
                modules.append(LowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn))
            else:
                modules.append(UpperNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn))
        
        modules.append(LinearSymplectic(sub_layers, dim))
        super(SympNet, self).__init__(*modules)

class LinearSympNet(LinearSymplectic, StepIntegrator):
    pass

class ConvLinearSympNet(nn.Sequential, StepIntegrator):
    def __init__(self, layers, dim):
        self.dim = dim

        dict = OrderedDict()

        for k in range(layers):
            if k % 2 == 0:
                dict['upper_conv1d_{}'.format(k)] = UpperSymplecticConv1d(dim, bias=k==(layers-1))
            else:
                dict['lower_conv1d_{}'.format(k)] = LowerSymplecticConv1d(dim, bias=k==(layers-1))

        super(ConvLinearSympNet, self).__init__(dict)
        

class HarmonicSympNet(nn.Sequential, StepIntegrator):
    def __init__(self, layers, sub_layers, dim, dt = 0.1):
        self.dim = dim
        modules = []

        for k in range(layers):
            modules.append(LinearSymplectic(sub_layers, dim))
            modules.append(HarmonicUnit(dt))
        
        modules.append(LinearSymplectic(sub_layers, dim))
        super(HarmonicSympNet, self).__init__(*modules)

# TODO implement fully connected layer
#class FNN(nn:Sequential):
#    def __init__():
#        pass