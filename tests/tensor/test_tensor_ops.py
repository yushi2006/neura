import unittest
import numpy as np
from neura import Tensor


class TestTensor(unittest.TestCase):
    def test_add(self):
        """Test addition of two Tensors with mixed requires_grad."""
        data1 = np.array([1, 2, 3], dtype=np.float32)
        data2 = np.array([4, 5, 6], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=False)
        result = tensor1 + tensor2
        expected_data = data1 + data2
        self.assertTrue(np.array_equal(result.data, expected_data))
        self.assertTrue(result.requires_grad)

    def test_add_no_grad(self):
        """Test addition when both Tensors have requires_grad=False."""
        data1 = np.array([1, 2, 3], dtype=np.float32)
        data2 = np.array([4, 5, 6], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=False)
        tensor2 = Tensor(data2, requires_grad=False)
        result = tensor1 + tensor2
        expected_data = data1 + data2
        self.assertTrue(np.array_equal(result.data, expected_data))
        self.assertFalse(result.requires_grad)

    def test_sub(self):
        """Test subtraction of two Tensors."""
        data1 = np.array([4, 5, 6], dtype=np.float32)
        data2 = np.array([1, 2, 3], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=True)
        result = tensor1 - tensor2
        expected_data = data1 - data2
        self.assertTrue(np.array_equal(result.data, expected_data))
        self.assertTrue(result.requires_grad)

    def test_mul(self):
        """Test multiplication of a Tensor by a scalar."""
        data = np.array([1, 2, 3], dtype=np.float32)
        scalar = np.float32(2.0)
        tensor = Tensor(data, requires_grad=True)
        result = tensor * scalar
        expected_data = data * scalar
        self.assertTrue(np.array_equal(result.data, expected_data))
        self.assertTrue(result.requires_grad)

    def test_mul_no_grad(self):
        """Test scalar multiplication with requires_grad=False."""
        data = np.array([1, 2, 3], dtype=np.float32)
        scalar = np.float32(2.0)
        tensor = Tensor(data, requires_grad=False)
        result = tensor * scalar
        expected_data = data * scalar
        self.assertTrue(np.array_equal(result.data, expected_data))
        self.assertFalse(result.requires_grad)

    def test_matmul(self):
        """Test matrix multiplication of two Tensors."""
        data1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=True)
        result = tensor1 @ tensor2
        expected_data = np.matmul(data1, data2)
        self.assertTrue(np.array_equal(result.data, expected_data))
        self.assertTrue(result.requires_grad)

    def test_matmul_mixed_grad(self):
        """Test matrix multiplication with mixed requires_grad settings."""
        data1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        # Case 1: First tensor requires grad
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=False)
        result = tensor1 @ tensor2
        self.assertTrue(result.requires_grad)
        # Case 2: Second tensor requires grad
        tensor3 = Tensor(data1, requires_grad=False)
        tensor4 = Tensor(data2, requires_grad=True)
        result2 = tensor3 @ tensor4
        self.assertTrue(result2.requires_grad)
        # Case 3: Neither requires grad
        tensor5 = Tensor(data1, requires_grad=False)
        tensor6 = Tensor(data2, requires_grad=False)
        result3 = tensor5 @ tensor6
        self.assertFalse(result3.requires_grad)

    def test_getitem(self):
        """Test indexing and slicing of a Tensor."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        tensor = Tensor(data)
        self.assertEqual(tensor[0], 1)
        self.assertEqual(tensor[2], 3)
        self.assertTrue(np.array_equal(tensor[1:4], np.array([2, 3, 4])))

    def test_getitem_type(self):
        """Test the types returned by indexing."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data)
        item = tensor[0]
        self.assertIsInstance(item, np.float32)
        slice_item = tensor[1:3]
        self.assertIsInstance(slice_item, np.ndarray)
        self.assertEqual(slice_item.dtype, np.float32)

    def test_repr(self):
        """Test the string representation of a Tensor."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data, dtype=np.float32)
        expected_repr = "[1. 2. 3.], dtype=<class 'numpy.float32'>"
        self.assertEqual(repr(tensor), expected_repr)

    def test_matmul_incompatible(self):
        """Test matrix multiplication with incompatible shapes."""
        data1 = np.array([[1, 2, 3]], dtype=np.float32)  # Shape (1, 3)
        data2 = np.array([[4, 5]], dtype=np.float32)  # Shape (1, 2)
        tensor1 = Tensor(data1)
        tensor2 = Tensor(data2)
        with self.assertRaises(ValueError):
            result = tensor1 @ tensor2

    def test_getitem_invalid(self):
        """Test indexing with an invalid index."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data)
        with self.assertRaises(IndexError):
            item = tensor[3]

    def test_iadd(self):
        """Test in-place addition of two Tensors."""
        data1 = np.array([1, 2, 3], dtype=np.float32)
        data2 = np.array([4, 5, 6], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=False)
        tensor1 += tensor2
        tensor3 = tensor1 + tensor2
        self.assertTrue(tensor1, tensor3)
        self.assertTrue(tensor1.requires_grad)

    def test_isub(self):
        """Test in-place subtraction of two Tensors."""
        data1 = np.array([4, 5, 6], dtype=np.float32)
        data2 = np.array([1, 2, 3], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=True)
        tensor1 -= tensor2
        tensor3 = tensor1 - tensor2
        self.assertTrue(tensor1, tensor3)
        self.assertTrue(tensor1.requires_grad)

    def test_imul(self):
        """Test in-place multiplication by a scalar."""
        data = np.array([1, 2, 3], dtype=np.float32)
        scalar = np.float32(2.0)
        tensor = Tensor(data, requires_grad=True)
        tensor *= scalar
        expected_tensor = tensor * scalar
        self.assertTrue(tensor, expected_tensor)
        self.assertTrue(tensor.requires_grad)

    def test_imatmul(self):
        """Test in-place matrix multiplication."""
        data1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        tensor1 = Tensor(data1, requires_grad=True)
        tensor2 = Tensor(data2, requires_grad=True)
        tensor1 @= tensor2
        expected_tensor = tensor1 @ tensor2
        self.assertTrue(tensor1, expected_tensor)
        self.assertTrue(tensor1.requires_grad)

    def test_len(self):
        """Test the __len__ method of Tensor."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data)
        self.assertEqual(len(tensor), 3)

        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(data)
        self.assertEqual(len(tensor), 4)

        data = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
        tensor = Tensor(data)
        self.assertEqual(len(tensor), 4)

        data = np.array([], dtype=np.float32)
        tensor = Tensor(data)
        self.assertEqual(len(tensor), 0)

    def test_view(self):
        """Test the view method of Tensor."""
        data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.view(2, 3)
        self.assertTrue(np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]])))
        self.assertTrue(tensor.requires_grad)

        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.view(4)
        self.assertTrue(np.array_equal(tensor.data, np.array([1, 2, 3, 4])))
        self.assertTrue(tensor.requires_grad)

        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.view(2, 2)
        self.assertTrue(np.array_equal(tensor.data, data))
        self.assertTrue(tensor.requires_grad)
        
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        tensor = Tensor(data)
        with self.assertRaises(ValueError):
            tensor.view(2, 3)

    def test_squeeze(self):
        """Test the squeeze method of Tensor."""
        data = np.array([[1, 2, 3]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.squeeze(0)
        self.assertTrue(np.array_equal(tensor.data, np.array([1, 2, 3])))
        self.assertTrue(tensor.requires_grad)

        data = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.squeeze(1)
        self.assertTrue(np.array_equal(tensor.data, np.array([[1, 2, 3], [4, 5, 6]])))
        self.assertTrue(tensor.requires_grad)

        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(data)
        with self.assertRaises(ValueError):
            tensor.squeeze(0)

    def test_unsqueeze(self):
        """Test the unsqueeze method of Tensor."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.unsqueeze(0)
        self.assertTrue(np.array_equal(tensor.data, np.array([[1, 2, 3]])))
        self.assertTrue(tensor.requires_grad)

        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        tensor.unsqueeze(1)
        expected = np.array([[[1, 2, 3]], [[4, 5, 6]]])
        self.assertTrue(np.array_equal(tensor.data, expected))
        self.assertTrue(tensor.requires_grad)

        tensor = Tensor(data, requires_grad=True)
        tensor.unsqueeze(2)
        expected = data[:, :, np.newaxis]  # shape (2,3,1)
        self.assertTrue(np.array_equal(tensor.data, expected))
        self.assertTrue(tensor.requires_grad)

        tensor = Tensor(data)
        with self.assertRaises(ValueError):
            tensor.unsqueeze(3)


if __name__ == "__main__":
    unittest.main()
