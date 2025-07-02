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
        self.assertIsNotNone(a, "The calculated gradient is None.")
        self.assertIsNotNone(b, "The expected gradient is None.")
        try:
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=tol)
        except AssertionError as e:
            self.fail(f"Arrays not almost equal:\nExpected: {b}\nGot:      {a}\n{e}")

    def test_add(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        z = x + y

        # Test backward on a non-scalar output
        z.backward()

        expected = np.ones_like(x.data)
        self.assertArrayAlmostEqual(x.grad, expected)
        self.assertArrayAlmostEqual(y.grad, expected)

    def test_sub(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = Tensor([5.0, 7.0], requires_grad=True)
        z = x - y

        z.backward()

        expected_x = np.ones_like(x.data)
        expected_y = -np.ones_like(y.data)
        self.assertArrayAlmostEqual(x.grad, expected_x)
        self.assertArrayAlmostEqual(y.grad, expected_y)

    def test_scalar_mul(self):
        scalar = 3.5
        x = Tensor([1.0, -2.0, 0.5], requires_grad=True)
        z = x * scalar

        z.backward()

        expected = np.ones_like(x.data) * scalar
        self.assertArrayAlmostEqual(x.grad, expected)

    def test_matmul(self):
        x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        y = Tensor(y_data, requires_grad=True)
        z = x @ y

        z.backward()  # Propagates a gradient of ones

        ones = np.ones_like(z.data)
        expected_x_grad = ones @ y_data.T
        expected_y_grad = x_data.T @ ones
        self.assertArrayAlmostEqual(x.grad, expected_x_grad)
        self.assertArrayAlmostEqual(y.grad, expected_y_grad)

    def test_chain_operations(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x + y
        scalar = 2.0
        w = z * scalar

        w.backward()

        expected = np.ones_like(x.data) * scalar
        self.assertArrayAlmostEqual(x.grad, expected)
        self.assertArrayAlmostEqual(y.grad, expected)

    def test_zero_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x + y

        z.backward()

        self.assertFalse(np.allclose(x.grad, 0))
        self.assertFalse(np.allclose(y.grad, 0))

        # Your zero_grad is recursive, so calling it on the result is correct.
        z.zero_grad()

        self.assertArrayAlmostEqual(x.grad, np.zeros_like(x.data))
        self.assertArrayAlmostEqual(y.grad, np.zeros_like(y.data))
        self.assertArrayAlmostEqual(z.grad, np.zeros_like(z.data))

    def test_conv2d(self):
        x_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        W_data = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        W = Tensor(W_data, requires_grad=True)
        z = x.conv2d(W)  # Output shape (1, 1, 2, 2)

        # The backward call will propagate a gradient of ones
        z.backward()

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
        z = x.conv2dTranspose(W, padding=0, stride=1)  # Output shape (1, 1, 3, 3)

        # Calling backward() on a non-scalar propagates a gradient of ones
        z.backward()

        # As derived above, dx is a convolution of the 3x3 output grad with the 2x2 kernel W
        expected_dx = np.array([[[[4, 4], [4, 4]]]], dtype=np.float32)

        # As derived above, dW is a convolution of the 2x2 input x with the 3x3 output grad
        expected_dW = np.array([[[[10, 10], [10, 10]]]], dtype=np.float32)

        self.assertArrayAlmostEqual(x.grad, expected_dx)
        self.assertArrayAlmostEqual(W.grad, expected_dW)

    def test_relu(self):
        data = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.relu().sum()

        out.backward()

        expected_grad = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.assertArrayAlmostEqual(t.grad, expected_grad)

    def test_log(self):
        data = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.log().sum()

        out.backward()

        expected_grad = np.array([1.0, 0.5, 0.25], dtype=np.float32)
        self.assertArrayAlmostEqual(t.grad, expected_grad)

    def test_exp(self):
        data = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.exp().sum()

        out.backward()

        expected_grad = np.exp(data).astype(np.float32)
        self.assertArrayAlmostEqual(t.grad, expected_grad)

    def test_abs(self):
        data = np.array([-3.0, 0.0, 4.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.abs().sum()

        out.backward()

        expected_grad = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        self.assertArrayAlmostEqual(t.grad, expected_grad)

    def test_sum(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.sum()

        out.backward()

        expected_grad = np.ones_like(data)
        self.assertArrayAlmostEqual(t.grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
