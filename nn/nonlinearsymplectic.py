from nn.linearsymplectic import SymplecticTriangularUnit
import torch
from torch import nn
import math

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
            nn.init.uniform_(self.a, -0.01, 0.01)
            nn.init.constant_(self.b, 0)
            nn.init.uniform_(self.K, -0.01, 0.01)
            
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + (self.activation_fn(x_bottom.mm(self.K.t()) + self.b)*self.a).mm(self.K)

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerGradientModule(UpperGradientModule):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom + (self.activation_fn(x_top.mm(self.K.t()) + self.b)*self.a).mm(self.K)

class NormalizedUpperGradientModule(SymplecticTriangularUnit):
    def __init__(self, dim:int, n:int, bias=False, activation_fn = torch.sigmoid):
        super(NormalizedUpperGradientModule, self).__init__(dim, bias, reset_params=False)
        self.activation_fn = activation_fn
        
        self.register_buffer('var', torch.ones(n))
        self.register_buffer('mean', torch.zeros(n))

        self.a = nn.Parameter(torch.Tensor(n))
        self.b = nn.Parameter(torch.Tensor(n))
        self.K = nn.Parameter(torch.Tensor(n, self.dim_half))
        self.gamma = nn.Parameter(torch.Tensor(n))
        self.beta = nn.Parameter(torch.Tensor(n))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

            nn.init.uniform_(self.a, -0.01, 0.01)
            nn.init.constant_(self.b, 0)
            nn.init.uniform_(self.K, -10, 10)

    def _normalize(self, input, eps = 1e-05):
        if self.training:
            self.var, self.mean = torch.var_mean(input, 0, unbiased=True)

        factor = self.gamma/torch.sqrt(self.var+eps)
        return factor*(input - self.mean) + self.beta, factor

    def _matrix_calc_top(self, x_top, x_bottom):
        pre_activation, factor = self._normalize(x_bottom.mm(self.K.t()) + self.b)
        return x_top + ((self.activation_fn(pre_activation)*factor*self.a).mm(self.K))

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class NormalizedLowerGradientModule(NormalizedUpperGradientModule):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top

    def _matrix_calc_bottom(self, x_top, x_bottom):
        pre_activation, factor = self._normalize(x_top.mm(self.K.t()) + self.b)
        return x_bottom + (self.activation_fn(pre_activation)*factor*self.a).mm(self.K)