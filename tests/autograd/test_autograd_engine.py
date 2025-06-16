import unittest
import numpy as np

# Adjust this import to match where your Tensor class is defined:
# For example, if your tensor code is in tensor.py in the same folder:
from neura import Tensor, Node

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
        x, y = Node(x), Node(y)
        z = x + y
        # Before backward, grads should be zero (or None)
        # (depending on your design; we only check after backward)
        z.backward()
        # For addition, dz/dx = 1, dz/dy = 1:
        expected = np.ones_like(x.tensor.data)
        self.assertArrayAlmostEqual(x.tensor.grad, expected)
        self.assertArrayAlmostEqual(y.tensor.grad, expected)
    
    def test_sub(self):
        # z = x - y
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = Tensor([5.0, 7.0], requires_grad=True)
        x, y = Node(x), Node(y)
        z = x - y
        z.backward()
        # dz/dx = 1, dz/dy = -1
        expected_x = np.ones_like(x.tensor.data)
        expected_y = -np.ones_like(y.tensor.data)
        self.assertArrayAlmostEqual(x.tensor.grad, expected_x)
        self.assertArrayAlmostEqual(y.tensor.grad, expected_y)

    def test_scalar_mul(self):
        # z = x * scalar
        scalar = 3.5
        x = Tensor([1.0, -2.0, 0.5], requires_grad=True)
        x = Node(x)
        z = x * scalar
        z.backward()
        # dz/dx = scalar
        expected = np.ones_like(x.tensor.data) * scalar
        self.assertArrayAlmostEqual(x.tensor.grad, expected)

    def test_matmul(self):
        # z = x @ y, where x: (2,2), y: (2,2)
        x_data = np.array([[1.0, 2.0],
                           [3.0, 4.0]], dtype=np.float32)
        y_data = np.array([[5.0, 6.0],
                           [7.0, 8.0]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        y = Tensor(y_data, requires_grad=True)
        x, y = Node(x), Node(y)
        z = x @ y  # shape (2,2)
        # Directly call backward on matrix output:
        z.backward()
        # Given your backward: z.grad is ones matrix of shape (2,2).
        # Then x.grad = ones(2,2) @ y.T
        ones = np.ones_like(z.tensor.data)
        expected_x_grad = ones @ y_data.T
        # y.grad = x.T @ ones
        expected_y_grad = x_data.T @ ones
        self.assertArrayAlmostEqual(x.tensor.grad, expected_x_grad)
        self.assertArrayAlmostEqual(y.tensor.grad, expected_y_grad)

    def test_chain_operations(self):
        # Test combination: w = (x + y) * scalar
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        x, y = Node(x), Node(y)
        z = x + y       # dz/dx = 1, dz/dy = 1
        scalar = 2.0
        w = z * scalar  # dw/dz = scalar
        w.backward()
        # dw/dx = scalar * dz/dx = scalar * 1
        expected = np.ones_like(x.tensor.data) * scalar
        self.assertArrayAlmostEqual(x.tensor.grad, expected)
        self.assertArrayAlmostEqual(y.tensor.grad, expected)

    def test_zero_grad(self):
        # Check that zero_grad resets gradients for a small graph
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        x, y = Node(x), Node(y)
        z = x + y
        z.backward()
        # Ensure grads are non-zero
        self.assertFalse(np.allclose(x.tensor.grad, 0))
        self.assertFalse(np.allclose(y.tensor.grad, 0))
        # Zero them via zero_grad on result
        z.zero_grad()
        # Now grads on x, y, z should be zero arrays
        self.assertArrayAlmostEqual(x.tensor.grad, np.zeros_like(x.tensor.data))
        self.assertArrayAlmostEqual(y.tensor.grad, np.zeros_like(y.tensor.data))
        self.assertArrayAlmostEqual(z.tensor.grad, np.zeros_like(z.tensor.data))

if __name__ == "__main__":
    unittest.main()
