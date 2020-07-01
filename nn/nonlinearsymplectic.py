from nn.linearsymplectic import UpperLinearSymplectic
import torch
from torch import sigmoid, nn

class UpperNonlinearSymplectic(UpperLinearSymplectic):
    def _matrix_calc_top(self, symmetric_matrix, x_top, x_bottom):
        return x_top + self.h*sigmoid(x_bottom)
    
    def _matrix_calc_bottom(self, symmetric_matrix, x_top, x_bottom):
        return x_bottom

class LowerNonlinearSymplectic(UpperLinearSymplectic):
    def _matrix_calc_top(self, symmetric_matrix, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, symmetric_matrix, x_top, x_bottom):
        return self.h*sigmoid(x_top) + x_bottom

# TODO only supports 2 dimensions at the moment.
class HarmonicUnit(nn.Module):
    def __init__(self, h):
        super(HarmonicUnit, self).__init__()
        self.h = h

        self.dt = nn.Parameter(torch.Tensor(1))
        self.m = nn.Parameter(torch.Tensor(1))
        self.omega = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.uniform_(self.dt, -self.h, self.h)
            nn.init.uniform_(self.m, 0, 1)
            nn.init.uniform_(self.omega, 0, 1)

    def forward(self, input):
        q0 = input[:,0]
        p0 = input[:,1]

        q = q0*torch.cos(self.omega*self.dt) + p0*torch.sin(self.omega*self.dt)
        p = -self.m*self.omega*q0*torch.sin(self.omega*self.dt) + p0*torch.cos(self.omega*self.dt)

        return torch.stack([q,p], dim=1)