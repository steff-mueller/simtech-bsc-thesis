import unittest
import math
import torch
from torch import sigmoid
from nn.nonlinearsymplectic import UpperNonlinearSymplectic, LowerNonlinearSymplectic, HarmonicUnit
from nn.symplecticloss import symplectic_mse_loss

class TestNonlinearSymplectic(unittest.TestCase):
    def test_upper(self):
        module = UpperNonlinearSymplectic(dim=4)

        p = torch.ones(2,1)
        q = torch.ones(2,1)
        input = torch.cat([p, q])

        bias = torch.tensor(
            [[1],
            [3],
            [6],
            [-3]],
            dtype=torch.float
        )

        module.bias.data = torch.reshape(bias, (-1,))

        expected = torch.cat([p + sigmoid(q), q]) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)
    
    def test_lower(self):
        module = LowerNonlinearSymplectic(dim=4)

        p = torch.ones(2,1)
        q = torch.ones(2,1)
        input = torch.cat([p, q])

        bias = torch.tensor(
            [[1],
            [3],
            [6],
            [-3]],
            dtype=torch.float
        )

        module.bias.data = torch.reshape(bias, (-1,))

        expected = torch.cat([p, sigmoid(p) + q]) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)

    def test_harmonic(self):
        h = 0.1
        module = HarmonicUnit(h)

        module.dt.data = torch.tensor(0.1)
        module.omega.data = torch.tensor(math.pi/h)
        module.m.data = torch.tensor(1.)

        input = torch.tensor([
            [0., 0.],
            [1., 2.]
        ], requires_grad=True)

        actual = module.forward(input)
        expected = torch.tensor([
            [0., 0.],
            [-1, -2]
        ])

        torch.testing.assert_allclose(symplectic_mse_loss(input, actual), 0)
        torch.testing.assert_allclose(actual, expected)     
     
if __name__ == '__main__':
    unittest.main()