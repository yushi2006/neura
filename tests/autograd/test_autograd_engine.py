import unittest

import numpy as np

# Adjust this import to match where your Tensor class is defined:
# For example, if your tensor code is in tensor.py in the same folder:
from neura import Tensor


class TestAutogradBackward(unittest.TestCase):
    def assertArrayAlmostEqual(self, a: np.ndarray, b: np.ndarray, tol: float = 1e-6):
        """
        Helper: assert that two numpy arrays are close.
        """
        try:
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=tol)
        except AssertionError as e:
            self.fail(f"Arrays not almost equal:\nExpected: {b}\nGot:      {a}\n{e}")

    def test_add(self):
        # z = x + y
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        # x, y = Node(x), Node(y)
        z = x + y
        # Before backward, grads should be zero (or None)
        # (depending on your design; we only check after backward)
        z.backward()
        # For addition, dz/dx = 1, dz/dy = 1:
        expected = np.ones_like(x.data)
        self.assertArrayAlmostEqual(x.grad, expected)
        self.assertArrayAlmostEqual(y.grad, expected)

    def test_sub(self):
        # z = x - y
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = Tensor([5.0, 7.0], requires_grad=True)
        # x, y = Node(x), Node(y)
        z = x - y
        z.backward()
        # dz/dx = 1, dz/dy = -1
        expected_x = np.ones_like(x.data)
        expected_y = -np.ones_like(y.data)
        self.assertArrayAlmostEqual(x.grad, expected_x)
        self.assertArrayAlmostEqual(y.grad, expected_y)

    def test_scalar_mul(self):
        # z = x * scalar
        scalar = 3.5
        x = Tensor([1.0, -2.0, 0.5], requires_grad=True)
        # x = Node(x)
        z = x * scalar
        z.backward()
        # dz/dx = scalar
        expected = np.ones_like(x.data) * scalar
        self.assertArrayAlmostEqual(x.grad, expected)

    def test_matmul(self):
        # z = x @ y, where x: (2,2), y: (2,2)
        x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        y = Tensor(y_data, requires_grad=True)
        # x, y = Node(x), Node(y)
        z = x @ y  # shape (2,2)
        # Directly call backward on matrix output:
        z.backward()
        # Given your backward: z.grad is ones matrix of shape (2,2).
        # Then x.grad = ones(2,2) @ y.T
        ones = np.ones_like(z.data)
        expected_x_grad = ones @ y_data.T
        # y.grad = x.T @ ones
        expected_y_grad = x_data.T @ ones
        self.assertArrayAlmostEqual(x.grad, expected_x_grad)
        self.assertArrayAlmostEqual(y.grad, expected_y_grad)

    def test_chain_operations(self):
        # Test combination: w = (x + y) * scalar
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        # x, y = Node(x), Node(y)
        z = x + y  # dz/dx = 1, dz/dy = 1
        scalar = 2.0
        w = z * scalar  # dw/dz = scalar
        w.backward()
        # dw/dx = scalar * dz/dx = scalar * 1
        expected = np.ones_like(x.data) * scalar
        self.assertArrayAlmostEqual(x.grad, expected)
        self.assertArrayAlmostEqual(y.grad, expected)

    def test_zero_grad(self):
        # Check that zero_grad resets gradients for a small graph
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        # x, y = Node(x), Node(y)
        z = x + y
        z.backward()
        # Ensure grads are non-zero
        self.assertFalse(np.allclose(x.grad, 0))
        self.assertFalse(np.allclose(y.grad, 0))
        # Zero them via zero_grad on result
        z.zero_grad()
        # Now grads on x, y, z should be zero arrays
        self.assertArrayAlmostEqual(x.grad, np.zeros_like(x.data))
        self.assertArrayAlmostEqual(y.grad, np.zeros_like(y.data))
        self.assertArrayAlmostEqual(z.grad, np.zeros_like(z.data))

    # New tests for conv2d and conv2dTranspose
    def test_conv2d(self):
        """Test gradients for 2D convolution."""
        x_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        W_data = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        W = Tensor(W_data, requires_grad=True)
        z = x.conv2d(W)
        grad_output = np.ones((1, 1, 2, 2), dtype=np.float32)
        z.backward(grad_output)
        expected_dx = np.array([[[[1, 1, 0], [1, 2, 1], [0, 1, 1]]]], dtype=np.float32)
        expected_dW = np.array([[[[12, 16], [24, 28]]]], dtype=np.float32)
        self.assertArrayAlmostEqual(x.grad, expected_dx)
        self.assertArrayAlmostEqual(W.grad, expected_dW)

    def test_conv2dTranspose(self):
        """Test gradients for transposed 2D convolution."""
        x_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        W_data = np.array([[[[1, 1], [1, 1]]]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        W = Tensor(W_data, requires_grad=True)
        z = x.conv2dTranspose(W, padding=0, stride=1)
        grad_output = np.ones((1, 1, 3, 3), dtype=np.float32)
        z.backward(grad_output)
        expected_dx = np.array([[[[4, 4], [4, 4]]]], dtype=np.float32)
        expected_dW = np.array([[[[10, 10], [10, 10]]]], dtype=np.float32)
        self.assertArrayAlmostEqual(x.grad, expected_dx)
        self.assertArrayAlmostEqual(W.grad, expected_dW)


if __name__ == "__main__":
    unittest.main()
