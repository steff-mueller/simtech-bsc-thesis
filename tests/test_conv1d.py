import unittest
import torch

from nn.linearsymplectic import (UpperSymplecticConv1d, LowerSymplecticConv1d,
    CanonicalSymmetricKernelBasis)

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

if __name__ == '__main__':
    unittest.main()