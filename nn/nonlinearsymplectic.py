from nn.linearsymplectic import SymplecticTriangularUnit
import torch
from torch import nn

class UpperNonlinearSymplectic(SymplecticTriangularUnit):
    def __init__(self, dim, bias=False, activation_fn = torch.sigmoid):
        super(UpperNonlinearSymplectic, self).__init__(dim, bias, reset_params=False)
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.Tensor(self.dim_half))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.normal_(self.a, 0., 0.01)

    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + self.a*self.activation_fn(x_bottom)
    
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerNonlinearSymplectic(UpperNonlinearSymplectic):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return self.a*self.activation_fn(x_top) + x_bottom