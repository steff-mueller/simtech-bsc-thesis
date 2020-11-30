import torch
from torch import nn
import numpy as np
from collections import OrderedDict

from nn.linearsymplectic import LinearSymplectic, LowerSymplecticConv1d, UpperSymplecticConv1d
from nn.nonlinearsymplectic import LowerNonlinearSymplectic, UpperNonlinearSymplectic

from models.vectors import NumpyPhaseSpace, PhaseSpaceVectorList

def integrate(model, q0, p0, t_start, t_end, dt, device=None, custom_phase_space=None):
    """
    Helper method to use neural networks as numerical time integrators.
    """

    dim_half = np.size(q0)
    dim = 2*dim_half

    phase_space = NumpyPhaseSpace(dim) if custom_phase_space is None else custom_phase_space
    result = PhaseSpaceVectorList()

    t_curr = t_start
    x_curr = torch.cat([
        torch.tensor(q0, dtype=torch.float32).reshape(1,dim_half), 
        torch.tensor(p0, dtype=torch.float32).reshape(1,dim_half), 
        ], dim=1).to(device)

    result.append(t_curr, phase_space.new_vector(q0, p0))
    
    # replicate behavior of original models 
    # and thus interpret t_end as an open interval
    while t_curr < t_end - dt:
        x_curr = model.__call__(x_curr)
        t_curr += dt

        # TODO why detach() required if original x_curr does not require grad?
        x_curr_numpy = x_curr.detach().numpy()
        q_curr = x_curr_numpy[0, 0:dim_half]
        p_curr = x_curr_numpy[0, dim_half:dim]
        result.append(t_curr, phase_space.new_vector(q_curr, p_curr))  

    return result