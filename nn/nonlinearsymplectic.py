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

class _NormBase(SymplecticTriangularUnit):
    norm_dim: int

    def __init__(self, dim: int, bias: bool, norm_dim: int, reset_params: bool, ignore_factor: bool):
        super(_NormBase, self).__init__(dim, bias, reset_params=False)
        self.norm_dim = norm_dim
        self.ignore_factor = ignore_factor

        self.register_buffer('var', torch.ones(self.norm_dim))
        self.register_buffer('mean', torch.zeros(self.norm_dim))

        self.gamma = nn.Parameter(torch.Tensor(self.norm_dim))
        self.beta = nn.Parameter(torch.Tensor(self.norm_dim))

        if reset_params:
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def _normalize(self, input, eps = 1e-05):
        if self.training:
            self.var, self.mean = torch.var_mean(input, 0, unbiased=True)

        factor = self.gamma/torch.sqrt(self.var+eps)
        return_factor = factor if not self.ignore_factor else 1
        return factor*(input - self.mean) + self.beta, return_factor

class NormalizedUpperNonlinearSymplectic(_NormBase):
    def __init__(self, dim, bias=False, activation_fn = torch.sigmoid, scalar_weight = False, ignore_factor=False):
        super(NormalizedUpperNonlinearSymplectic, self).__init__(
            dim, bias, norm_dim=int(dim/2), reset_params=False, ignore_factor=ignore_factor)
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.Tensor(1 if scalar_weight else self.dim_half))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.normal_(self.a, 0., 0.01)

    def _matrix_calc_top(self, x_top, x_bottom):
        x_bottom_normalized, factor = self._normalize(x_bottom)
        return x_top + self.a*factor*self.activation_fn(x_bottom_normalized)
    
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class NormalizedLowerNonlinearSymplectic(NormalizedUpperNonlinearSymplectic):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, x_top, x_bottom):
        x_top_normalized, factor = self._normalize(x_top)
        return self.a*factor*self.activation_fn(x_top_normalized) + x_bottom

class NormalizedUpperGradientModule(_NormBase):
    def __init__(self, dim:int, n:int, bias=False, activation_fn = torch.sigmoid, ignore_factor=False):
        super(NormalizedUpperGradientModule, self).__init__(
            dim, bias, norm_dim=n, reset_params=False, ignore_factor=ignore_factor)
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
            nn.init.uniform_(self.K, -10, 10)

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

class UpperConv1dGradientModule(SymplecticTriangularUnit):
    def __init__(self, dim:int, n:int, bias=False, activation_fn = torch.sigmoid):
        super(UpperConv1dGradientModule, self).__init__(dim, bias, reset_params=False)
        self.n = n
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.Tensor(self.n))
        self.b = nn.Parameter(torch.Tensor(self.n))
        self.K = nn.Parameter(torch.Tensor(1,1,self.n))
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.uniform_(self.a, -0.01, 0.01)
            nn.init.constant_(self.b, 0)
            nn.init.uniform_(self.K, -0.01, 0.01)

    def _conv1d(self, x_half: torch.Tensor):
        x_half = x_half.reshape((-1, 1, self.dim_half))
        pre_activation = nn.functional.conv_transpose1d(x_half, self.K, stride=self.n)
        pre_activation += self.b.expand((self.dim_half,self.n)).reshape((1,self.dim_half*self.n))
        activation = self.activation_fn(pre_activation)
        result = nn.functional.conv1d(activation, self.K*self.a, stride=self.n)
        return result.reshape((-1,self.dim_half))

    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + self._conv1d(x_bottom)

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerConv1dGradientModule(UpperConv1dGradientModule):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return self._conv1d(x_top) + x_bottom

class NormalizedUpperConv1dGradientModule(_NormBase):
    def __init__(self, dim:int, n:int, bias=False, activation_fn = torch.sigmoid, ignore_factor=False):
        super(NormalizedUpperConv1dGradientModule, self).__init__(
            dim, bias, norm_dim=n, reset_params=False, ignore_factor=ignore_factor)
        self.n = n
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.Tensor(self.n))
        self.b = nn.Parameter(torch.Tensor(self.n))
        self.K = nn.Parameter(torch.Tensor(1,1,self.n))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.uniform_(self.a, -0.01, 0.01)
            nn.init.uniform_(self.b, -0.01, 0.01)
            nn.init.uniform_(self.K, -10, 10)

    def _conv1d(self, x_half: torch.Tensor):
        x_half = x_half.reshape((-1, 1, self.dim_half))
        pre_activation = nn.functional.conv_transpose1d(x_half, self.K, stride=self.n)
        pre_activation += self.b.expand((self.dim_half,self.n)).reshape((1,self.dim_half*self.n))

        n_training = pre_activation.shape[0]
        pre_activation_normalized, factor = self._normalize(pre_activation.reshape((n_training*self.dim_half,self.n)))
        pre_activation_normalized = pre_activation_normalized.reshape((n_training,1,self.dim_half*self.n))

        activation = self.activation_fn(pre_activation_normalized)
        result = nn.functional.conv1d(activation, self.K*factor*self.a, stride=self.n)
        return result.reshape((-1,self.dim_half))

    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + self._conv1d(x_bottom)

    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom
        
class NormalizedLowerConv1dGradientModule(NormalizedUpperConv1dGradientModule):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return self._conv1d(x_top) + x_bottom