import unittest
import torch

from nn.linearsymplectic import (UpperSymplecticConv1d, LowerSymplecticConv1d,
    CanonicalSymmetricKernelBasis)
from nn.nonlinearsymplectic import UpperConv1dGradientModule

class TestConv1d(unittest.TestCase):
    def test_upper(self):
        basis = CanonicalSymmetricKernelBasis(3)
        module = UpperSymplecticConv1d(dim=8, bias=False, kernel_basis=basis)

        k1 = 2
        k2 = -1
        matrix = torch.tensor([
            [1, 0, 0, 0, k1, k2, 0,  0],
            [0, 1, 0, 0, k2, k1, k2, 0],
            [0, 0, 1, 0, 0,  k2, k1, k2],
            [0, 0, 0, 1, 0,  0,  k2, k1],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=torch.float)

        input1 = torch.arange(8, dtype=torch.float)
        input2 = torch.arange(8, dtype=torch.float) + 5
        input = torch.stack([input1, input2])

        module.a.data = torch.tensor([k1, k2], dtype=torch.float).reshape(2,1)

        expected1 = matrix.mm(input1.reshape(8,1)).reshape(1,8)
        expected2 = matrix.mm(input2.reshape(8,1)).reshape(1,8)
        expected = torch.cat([expected1, expected2])

        actual = module.forward(input)

        self.assertEqual(actual.shape[0], 2)
        self.assertEqual(actual.shape[1], 8)
        torch.testing.assert_allclose(actual, expected)

    def test_lower(self):
        basis = CanonicalSymmetricKernelBasis(3)
        module = LowerSymplecticConv1d(dim=8, bias=False, kernel_basis=basis)

        k1 = 2
        k2 = -1
        matrix = torch.tensor([
            [1, 0, 0, 0, k1, k2, 0,  0],
            [0, 1, 0, 0, k2, k1, k2, 0],
            [0, 0, 1, 0, 0,  k2, k1, k2],
            [0, 0, 0, 1, 0,  0,  k2, k1],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=torch.float).t()

        input1 = torch.arange(8, dtype=torch.float)
        input2 = torch.arange(8, dtype=torch.float) + 5
        input = torch.stack([input1, input2])

        module.a.data = torch.tensor([k1, k2], dtype=torch.float).reshape(2,1)

        expected1 = matrix.mm(input1.reshape(8,1)).reshape(1,8)
        expected2 = matrix.mm(input2.reshape(8,1)).reshape(1,8)
        expected = torch.cat([expected1, expected2])

        actual = module.forward(input)

        self.assertEqual(actual.shape[0], 2)
        self.assertEqual(actual.shape[1], 8)
        torch.testing.assert_allclose(actual, expected)

    def test_upper_nonlinear(self):
        module = UpperConv1dGradientModule(dim = 8, n=10, bias=False)
        
        kernel = torch.arange(1, 11, dtype=torch.float)
        conv_matrix = torch.zeros(4,40, dtype=torch.float)
        conv_matrix[0,0:10] = kernel
        conv_matrix[1,10:20] = kernel
        conv_matrix[2,20:30] = kernel
        conv_matrix[3,30:40] = kernel

        transposed_conv_matrix = conv_matrix.t()

        a = torch.arange(-5,5, dtype=torch.float)
        a_expanded = a.expand(4,10).reshape(40)

        b = torch.arange(0,10)
        b_expanded = b.expand(4,10).reshape(40,1)

        module.a.data = a
        module.b.data = b
        module.K.data = kernel.reshape(1,1,10)

        torch.manual_seed(0)
        q = torch.rand(4,2, dtype=torch.float)
        p = torch.rand(4,2, dtype=torch.float)
        input = torch.cat([q,p], dim=0)

        expected_q = q + conv_matrix.mm(
            torch.diag(a_expanded).mm(
                torch.sigmoid(transposed_conv_matrix.mm(p) + b_expanded
            )))
        
        expected = torch.cat([expected_q, p], dim=0)

        actual = module.forward(input.t())

        self.assertEqual(actual.shape, (2,8))
        torch.testing.assert_allclose(actual.t(), expected)

if __name__ == '__main__':
    unittest.main()