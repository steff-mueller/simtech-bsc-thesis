import unittest
import torch
from nn.linearsymplectic import UpperLinearSymplectic, LowerLinearSymplectic, LinearSymplectic

class TestLinearSympletic(unittest.TestCase):
    def test_upper(self):
        module = UpperLinearSymplectic(dim=4)

        matrix = torch.tensor(
            [[1, 0, 4, 7],
            [0, 1, 7, 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
            dtype=torch.float
        )

        input = torch.ones(4,1)

        bias = torch.tensor(
            [[1],
            [3],
            [6],
            [-3]],
            dtype=torch.float
        )

        module.S.data = torch.tensor(
            [[2,3],
            [4,1]],
            dtype=torch.float
        )

        module.bias.data = torch.reshape(bias, (-1,))

        expected = matrix.mm(input) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)

    def test_lower(self):
        h = 0.1
        module = LowerLinearSymplectic(dim=4)

        matrix = torch.tensor(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [4, 7, 1, 0],
            [7, 2, 0, 1]],
            dtype=torch.float
        )

        input = torch.ones(4,1)

        bias = torch.tensor(
            [[1],
            [3],
            [6],
            [-3]],
            dtype=torch.float
        )

        module.S.data = torch.tensor(
            [[2,3],
            [4,1]],
            dtype=torch.float
        )

        module.bias.data = torch.reshape(bias, (-1,))

        expected = matrix.mm(input) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)

    def test_linear(self):
        h = 0.1
        module = LinearSymplectic(n=2, dim=4)

        matrix1 = torch.tensor(
            [[1, 0, 4, 7],
            [0, 1, 7, 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
            dtype=torch.float
        )

        matrix2 = torch.tensor(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [4, 7, 1, 0],
            [7, 2, 0, 1]],
            dtype=torch.float
        )

        bias = torch.tensor(
            [[1],
            [3],
            [6],
            [-3]],
            dtype=torch.float
        ) 

        input = torch.ones(4,1)

        module[0].S.data = module[1].S.data = torch.tensor(
            [[2,3],
            [4,1]],
            dtype=torch.float
        )
        module[1].bias.data = torch.reshape(bias, (-1,))

        expected = matrix2.mm(matrix1.mm(input)) + bias
        actual = module.forward(torch.reshape(input, (1,4)))
        
        torch.testing.assert_allclose(torch.reshape(actual, (4,1)), expected)

if __name__ == '__main__':
    unittest.main()