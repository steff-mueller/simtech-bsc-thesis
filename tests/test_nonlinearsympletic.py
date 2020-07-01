import unittest
import math
import torch
from torch import sigmoid
from nn.nonlinearsymplectic import UpperNonlinearSymplectic, LowerNonlinearSymplectic, HarmonicUnit

class TestNonlinearSymplectic(unittest.TestCase):
    def test_upper(self):
        h = 0.1
        module = UpperNonlinearSymplectic(d=2, h=h)

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

        module.S.data = torch.tensor(
            [[2,3],
            [3,1]]
        )

        module.bias.data = torch.reshape(bias, (-1,))

        expected = torch.cat([p + h*sigmoid(q), q]) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)
    
    def test_lower(self):
        h = 0.1
        module = LowerNonlinearSymplectic(d=2, h=h)

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

        module.S.data = torch.tensor(
            [[2,3],
            [3,1]]
        )

        module.bias.data = torch.reshape(bias, (-1,))

        expected = torch.cat([p, h*sigmoid(p) + q]) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)

    def test_harmonic(self):
        h = 0.1
        module = HarmonicUnit(h)

        module.omega.data = torch.tensor(math.pi/h)
        module.m.data = torch.tensor(1.)

        input = torch.tensor([
            [0., 0.],
            [1., 2.]
        ])

        actual = module.forward(input)
        expected = torch.tensor([
            [0., 0.],
            [-1, -2]
        ])

        torch.testing.assert_allclose(actual, expected)

if __name__ == '__main__':
    unittest.main()