import unittest
import math
import torch
from torch import sigmoid
from nn.nonlinearsymplectic import (UpperNonlinearSymplectic, LowerNonlinearSymplectic, 
    UpperGradientModule, LowerGradientModule)
from nn.symplecticloss import symplectic_mse_loss

class TestNonlinearSymplectic(unittest.TestCase):
    def test_upper(self):
        module = UpperNonlinearSymplectic(dim=4, bias=True)
        module.a.data = torch.ones(2)

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
        module = LowerNonlinearSymplectic(dim=4, bias=True)
        module.a.data = torch.ones(2)

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

    def test_lower_gradient_module(self):
        module = LowerGradientModule(dim=4, n=5, bias=False, activation_fn=torch.sigmoid)

        x1 = torch.tensor([3, 5, -1, 2], dtype=torch.float32).reshape((4,1))
        q1 = x1[0:2]
        p1 = x1[2:4]

        x2 = torch.tensor([-7, 7, 6, 3], dtype=torch.float32).reshape((4,1))
        q2 = x2[0:2]
        p2 = x2[2:4]

        K = torch.tensor([
            [-1, 2],
            [0, 3],
            [5, 4.1],
            [4, 1],
            [9, -7]
        ], dtype=torch.float32)

        a = torch.tensor([0.7, 2, 5, 6, 1], dtype=torch.float32)
        b = torch.tensor([3, 4, 1, 2, -1], dtype=torch.float32).reshape((5,1))

        Q1 = q1
        P1 = K.t().mm(torch.diag(a).mm(torch.sigmoid(K.mm(q1) + b))) + p1
        X1 = torch.cat([Q1, P1], dim=0)

        Q2 = q2
        P2 = K.t().mm(torch.diag(a).mm(torch.sigmoid(K.mm(q2) + b))) + p2
        X2 = torch.cat([Q2, P2], dim=0)

        module.K.data = K
        module.b.data = b.flatten()
        module.a.data = a
        expected = torch.cat([X1, X2], dim=1).t()

        actual = module.forward(torch.cat([x1, x2], dim=1).t())
        self.assertEqual(actual.shape, (2,4))
        torch.testing.assert_allclose(actual, expected)

    def test_upper_gradient_module(self):
        module = UpperGradientModule(dim=4, n=5, bias=False, activation_fn=torch.sigmoid)

        x1 = torch.tensor([3, 5, -1, 2], dtype=torch.float32).reshape((4,1))
        q1 = x1[0:2]
        p1 = x1[2:4]

        x2 = torch.tensor([-7, 7, 6, 3], dtype=torch.float32).reshape((4,1))
        q2 = x2[0:2]
        p2 = x2[2:4]

        K = torch.tensor([
            [-1, 2],
            [0, 3],
            [5, 4.1],
            [4, 1],
            [9, -7]
        ], dtype=torch.float32)

        a = torch.tensor([0.7, 2, 5, 6, 1], dtype=torch.float32)
        b = torch.tensor([3, 4, 1, 2, -1], dtype=torch.float32).reshape((5,1))

        Q1 = K.t().mm(torch.diag(a).mm(torch.sigmoid(K.mm(p1) + b))) + q1
        P1 = p1
        X1 = torch.cat([Q1, P1], dim=0)

        Q2 = K.t().mm(torch.diag(a).mm(torch.sigmoid(K.mm(p2) + b))) + q2
        P2 = p2
        X2 = torch.cat([Q2, P2], dim=0)

        module.K.data = K
        module.b.data = b.flatten()
        module.a.data = a
        expected = torch.cat([X1, X2], dim=1).t()

        actual = module.forward(torch.cat([x1, x2], dim=1).t())
        self.assertEqual(actual.shape, (2,4))
        torch.testing.assert_allclose(actual, expected)
   
if __name__ == '__main__':
    unittest.main()