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
            # Inspired by initialization of Linear layers:
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            nn.init.uniform_(self.a, -math.sqrt(1/self.dim_half), math.sqrt(1/self.dim_half))
            nn.init.uniform_(self.b, -math.sqrt(1/self.dim_half), math.sqrt(1/self.dim_half))
            nn.init.uniform_(self.K, -math.sqrt(1/self.dim_half), math.sqrt(1/self.dim_half))
            
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + (self.activation_fn(x_bottom.mm(self.K.t()) + self.b)*self.a).mm(self.K)

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerGradientModule(UpperGradientModule):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom + (self.activation_fn(x_top.mm(self.K.t()) + self.b)*self.a).mm(self.K)

class NormalizedUpperGradientModule(UpperGradientModule):
    def __init__(self, dim:int, n:int, bias=False, activation_fn = torch.sigmoid, affine=True):
        super(NormalizedUpperGradientModule, self).__init__(dim, n, bias, activation_fn)
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.Tensor(self.dim_half))
            self.beta = nn.Parameter(torch.Tensor(self.dim_half))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        self.register_buffer('var', torch.ones(self.dim_half))
        self.register_buffer('mean', torch.zeros(self.dim_half))
        self.reset_parameters2()

    # TODO refactor
    def reset_parameters2(self):
        if self.affine:
            with torch.no_grad():
                nn.init.ones_(self.gamma)
                nn.init.zeros_(self.beta)

    def _matrix_calc_top(self, x_top, x_bottom):
        if self.training:
            self.var, self.mean = torch.var_mean(x_bottom, 0, unbiased=True)

        eps = 1e-05
        gamma = self.gamma if self.affine else 1
        beta = self.beta if self.affine else 0
        factor = gamma/torch.sqrt(self.var+eps)

        x_bottom_normalized =  factor*(x_bottom - self.mean) + beta
        return factor*super()._matrix_calc_top(x_top, x_bottom_normalized)