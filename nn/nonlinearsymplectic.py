from nn.linearsymplectic import SymplecticTriangularUnit
import torch
from torch import nn

class UpperNonlinearSymplectic(SymplecticTriangularUnit):
    def __init__(self, dim, bias=False, activation_fn = torch.sigmoid, scalar_weight = False):
        super(UpperNonlinearSymplectic, self).__init__(dim, bias, reset_params=False)
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.Tensor(1 if scalar_weight else self.dim_half))
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

class UpperGradientModule(SymplecticTriangularUnit):
    def __init__(self, dim:int, n:int, bias=False, activation_fn = torch.sigmoid):
        super(UpperGradientModule, self).__init__(dim, bias, reset_params=False)
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.Tensor(n))
        self.b = nn.Parameter(torch.Tensor(n))
        self.K = nn.Parameter(torch.Tensor(n, self.dim_half))
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.constant_(self.a, 1)
            nn.init.constant_(self.b, 0)
            nn.init.eye_(self.K)
            
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + (self.activation_fn(x_bottom.mm(self.K.t()) + self.b)*self.a).mm(self.K)

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerGradientModule(UpperGradientModule):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom + (self.activation_fn(x_top.mm(self.K.t()) + self.b)*self.a).mm(self.K)