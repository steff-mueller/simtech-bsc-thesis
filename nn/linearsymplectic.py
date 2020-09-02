import torch
from torch import nn

import math
from collections import OrderedDict
from abc import ABC, abstractmethod

class SymplecticTriangularUnit(nn.Module, ABC):
    def __init__(self, dim, bias=True, reset_params=True):
        assert dim%2 == 0, 'Dimension of phase space must be even.'
        super(SymplecticTriangularUnit, self).__init__()
        self.dim = dim
        self.dim_half = int(dim/2)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.dim))
        else:
            self.register_parameter('bias', None)

        if reset_params:
            self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)

    def forward(self, input):
        x_top = input[:, 0:self.dim_half]
        x_bottom = input[:, self.dim_half:self.dim]

        result_top = self._matrix_calc_top(x_top, x_bottom)
        result_bottom = self._matrix_calc_bottom(x_top, x_bottom)

        result = torch.cat([result_top, result_bottom], dim=1)
        if self.bias is not None:
            result = result + self.bias

        return result

    @abstractmethod
    def _matrix_calc_top(self, x_top, x_bottom):
        pass

    @abstractmethod
    def _matrix_calc_bottom(self, x_top, x_bottom):
        pass

class UpperLinearSymplectic(SymplecticTriangularUnit):
    def __init__(self, dim, bias=True):
        super(UpperLinearSymplectic, self).__init__(dim, bias, reset_params=False)
        self.S = nn.Parameter(torch.Tensor(int(dim/2), int(dim/2)))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.normal_(self.S, 0., 0.01)

    def forward(self, input):
        self.symmetric_matrix = self.S + self.S.t()
        return super().forward(input)

    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + x_bottom.mm(self.symmetric_matrix)
    
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerLinearSymplectic(UpperLinearSymplectic):
   def _matrix_calc_top(self, x_top, x_bottom):
       return x_top
   
   def _matrix_calc_bottom(self, x_top, x_bottom):
       return x_top.mm(self.symmetric_matrix) + x_bottom

class LinearSymplectic(nn.Sequential):
    def __init__(self, n, dim, bias=True):
        dict = OrderedDict()

        # add upper and lower linear symplectic unit alternately
        for i in range(n):
            is_last = (i+1) == n and bias # only add bias to last linear symplectic unit.
            if (i % 2 == 0):
                dict['upper_linear_symp{}'.format(i)] = UpperLinearSymplectic(dim, bias=is_last)
            else:
                dict['lower_linear_symp{}'.format(i)] = LowerLinearSymplectic(dim, bias=is_last)

        super(LinearSymplectic, self).__init__(dict)

class UpperSymplecticConv1d(SymplecticTriangularUnit):
    def __init__(self, dim, bias=True, kernel_size = 3):
        super(UpperSymplecticConv1d, self).__init__(dim, bias, reset_params=False)
        assert kernel_size % 2 == 1, 'Kernel size must be odd.'
        self.kernel_size = kernel_size
        self.k = nn.Parameter(torch.Tensor(kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            nn.init.normal_(self.k, 0, 0.01)

    def conv1d_(self, x_half):
        n = x_half.shape[0] 
        x_half = x_half.reshape((n, 1, self.dim_half))

        kernel = (self.k+self.k.flip(0)).true_divide(2).reshape((1,1,self.kernel_size))

        conv_result = nn.functional.conv1d(x_half, kernel, padding=int(self.kernel_size/2))
        conv_result = conv_result.reshape(n, self.dim_half)

        return conv_result

    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top + self.conv1d_(x_bottom)
    
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom

class LowerSymplecticConv1d(UpperSymplecticConv1d):
    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return self.conv1d_(x_top) + x_bottom

class SymplecticScaling(SymplecticTriangularUnit):
    def __init__(self, dim, a_init):
        super(SymplecticScaling, self).__init__(dim, bias=False, reset_params=False)
        self.a_init = a_init
        self.a = nn.Parameter(torch.Tensor(self.dim_half))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            self.a.data = self.a_init

    def _matrix_calc_top(self, x_top, x_bottom):
        return x_top.mul(self.a)
   
    def _matrix_calc_bottom(self, x_top, x_bottom):
        return x_bottom.div(self.a)