import torch
from torch import nn

# l_up for now
class LinearSymplectic(nn.Module):
    def __init__(self, d, h, bias=True):
        super(LinearSymplectic, self).__init__()
        self.d = d
        self.h = h

        self.S = nn.Parameter(torch.Tensor(d, d))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(2*d))
        else:
            self.register_parameter('bias', None)

    # from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L79
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.S, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        symmetric_matrix = (self.S + self.S.t())/2.

        x_top = input[0:self.d]
        x_bottom = input[self.d:2*self.d]

        result_top = x_top + symmetric_matrix.mm(x_bottom)
        result_bottom = x_bottom

        return torch.cat([result_top, result_bottom]) + self.bias