import unittest
import torch
from torch import sigmoid
from nonlinearsymplectic import UpperNonlinearSymplectic, LowerNonlinearSymplectic

class TestNonlinearSymplectic(unittest.TestCase):
    def test_upper(self):
        module = UpperNonlinearSymplectic(d=2, h=0.1)

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

        expected = torch.cat([p + sigmoid(q), q]) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)
    
    def test_lower(self):
        module = LowerNonlinearSymplectic(d=2, h=0.1)

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

        expected = torch.cat([p, sigmoid(p) + q]) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)

if __name__ == '__main__':
    unittest.main()