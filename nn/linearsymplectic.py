import torch
from torch import nn

import math
from collections import OrderedDict

class UpperLinearSymplectic(nn.Module):
    def __init__(self, d, h, bias=True):
        super(UpperLinearSymplectic, self).__init__()
        self.d = d
        self.h = h

        self.S = nn.Parameter(torch.Tensor(d, d))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(2*d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.S, 0., 0.01)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0)

    def forward(self, input):
        symmetric_matrix = self.h*(self.S + self.S.t())

        x_top = input[:, 0:self.d]
        x_bottom = input[:, self.d:2*self.d]

        result_top = self._matrix_calc_top(symmetric_matrix, x_top, x_bottom)
        result_bottom = self._matrix_calc_bottom(symmetric_matrix, x_top, x_bottom)

        result = torch.cat([result_top, result_bottom], dim=1)
        if self.bias is not None:
            result = result + self.bias

        return result

    def _matrix_calc_top(self, symmetric_matrix, x_top, x_bottom):
        return x_top + x_bottom.mm(symmetric_matrix)
    
    def _matrix_calc_bottom(self, symmetric_matrix, x_top, x_bottom):
        return x_bottom

class LowerLinearSymplectic(UpperLinearSymplectic):
   def _matrix_calc_top(self, symmetric_matrix, x_top, x_bottom):
       return x_top
   
   def _matrix_calc_bottom(self, symmetric_matrix, x_top, x_bottom):
       return x_top.mm(symmetric_matrix) + x_bottom

class LinearSymplectic(nn.Sequential):
    def __init__(self, n, d, h, bias=True):
        dict = OrderedDict()

        # add upper and lower linear symplectic unit alternately
        for i in range(n):
            is_last = (i+1) == n and bias # only add bias to last linear symplectic unit.
            if (i % 2 == 0):
                dict['upper_linear_symp{}'.format(i)] = UpperLinearSymplectic(d, h, bias=is_last)
            else:
                dict['lower_linear_symp{}'.format(i)] = LowerLinearSymplectic(d, h, bias=is_last)

        super(LinearSymplectic, self).__init__(dict)