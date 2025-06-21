import unittest

import neura
import numpy as np
from neura.nn import Linear  # Assuming Linear is defined in my_module


class TestLinear(unittest.TestCase):
    def test_initialization(self):
        """Test that weights and biases are initialized with correct shapes and values."""
        layer = Linear(2, 3)
        self.assertEqual(layer.W.data.shape, (2, 3))
        self.assertEqual(layer.b.data.shape, (3,))
        self.assertTrue(np.all(layer.b.data == 0))

    def test_forward(self):
        """Test the forward pass with different batch sizes."""
        np.random.seed(42)  # For reproducibility
        layer = Linear(2, 3)

        # Test with batch size 2
        x1 = neura.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        output1 = layer.forward(x1)
        expected_output1 = np.dot(x1.data, layer.W.data) + layer.b.data
        self.assertTrue(np.allclose(output1.data, expected_output1))
        self.assertEqual(output1.data.shape, (2, 3))

        # Test with batch size 1
        x2 = neura.Tensor(np.array([[5.0, 6.0]]))
        output2 = layer.forward(x2)
        expected_output2 = np.dot(x2.data, layer.W.data) + layer.b.data
        self.assertTrue(np.allclose(output2.data, expected_output2))
        self.assertEqual(output2.data.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
