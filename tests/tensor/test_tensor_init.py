import unittest
import numpy as np
from neura import Tensor

class TestTensorInitStrategies(unittest.TestCase):
    def test_init(self):
        """Test Tensor initialization with default parameters."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True, dtype=np.float32)
        self.assertTrue(np.array_equal(tensor.data, data))
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float32)
    
    def test_ones(self):
        # Default requires_grad and dtype
        tensor = Tensor.ones((2, 3))
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertTrue(np.array_equal(tensor.data, np.ones((2, 3), dtype=np.float32)))
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float32)

        # Specified requires_grad and dtype
        tensor = Tensor.ones((3,), requires_grad=False, dtype=np.float64)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (3,))
        self.assertTrue(np.array_equal(tensor.data, np.ones((3,), dtype=np.float64)))
        self.assertFalse(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float64)

        # Scalar tensor
        tensor = Tensor.ones(())
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor.data, 1.0)
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float32)

    def test_zeros(self):
        # Default requires_grad and dtype
        tensor = Tensor.zeros((2, 2))
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (2, 2))
        self.assertTrue(np.array_equal(tensor.data, np.zeros((2, 2), dtype=np.float32)))
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float32)

        # Specified requires_grad and dtype
        tensor = Tensor.zeros((1, 4), requires_grad=False, dtype=np.int32)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (1, 4))
        self.assertTrue(np.array_equal(tensor.data, np.zeros((1, 4), dtype=np.int32)))
        self.assertFalse(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.int32)

        # Scalar tensor
        tensor = Tensor.zeros(())
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor.data, 0.0)
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float32)

    def test_randn(self):
        # Set seed for reproducibility
        np.random.seed(42)

        # Default requires_grad and dtype
        tensor = Tensor.randn((1000,))
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (1000,))
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float32)

        # Specified requires_grad and dtype
        np.random.seed(42)
        tensor = Tensor.randn((1000,), requires_grad=False, dtype=np.float64)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (1000,))
        self.assertFalse(tensor.requires_grad)
        self.assertEqual(tensor.dtype, np.float64)

if __name__ == "__main__":
    unittest.main()