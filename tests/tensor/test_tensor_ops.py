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
        self.assertEqual(tensor[0].data, 1.0)
        self.assertEqual(tensor[2].data, 3.0)
        self.assertTrue(np.array_equal(tensor[1:4], Tensor(np.array([2, 3, 4]))))

    def test_getitem_type(self):
        """Test the types returned by indexing."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(data)
        item = tensor[0]
        self.assertIsInstance(item, Tensor)
        slice_item = tensor[1:3]
        self.assertIsInstance(slice_item, Tensor)
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

    def test_conv2d_basic(self):
        """Test basic 2D convolution with known input and kernel."""
        # Input: (1, 1, 3, 3) - 1 batch, 1 channel, 3x3 spatial dimensions
        input_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
        # Kernel: (1, 1, 2, 2) - 1 output channel, 1 input channel, 2x2 kernel
        kernel_data = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        x = Tensor(input_data, requires_grad=True)
        kernel = Tensor(kernel_data, requires_grad=True)
        output = x.conv2d(kernel, padding=0, stride=1)
        # Expected output: (1, 1, 2, 2)
        # Computed manually:
        # [1*1 + 2*0 + 4*0 + 5*1, 2*1 + 3*0 + 5*0 + 6*1] = [6, 8]
        # [4*1 + 5*0 + 7*0 + 8*1, 5*1 + 6*0 + 8*0 + 9*1] = [12, 14]
        expected_data = np.array([[[[6, 8], [12, 14]]]], dtype=np.float32)
        self.assertTrue(np.allclose(output.data, expected_data))
        self.assertEqual(output.shape, (1, 1, 2, 2))
        self.assertTrue(output.requires_grad)

    def test_conv2d_stride(self):
        """Test 2D convolution with stride > 1."""
        # Input: (1, 1, 4, 4) - 1 batch, 1 channel, 4x4 spatial dimensions
        input_data = np.array(
            [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
            dtype=np.float32,
        )
        # Kernel: (1, 1, 2, 2)
        kernel_data = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        x = Tensor(input_data, requires_grad=True)
        kernel = Tensor(kernel_data, requires_grad=True)
        output = x.conv2d(kernel, stride=(2, 2))
        # Expected output: (1, 1, 2, 2)
        # With stride=2:
        # [1*1 + 2*0 + 5*0 + 6*1, 3*1 + 4*0 + 7*0 + 8*1] = [7, 11]
        # [9*1 + 10*0 + 13*0 + 14*1, 11*1 + 12*0 + 15*0 + 16*1] = [23, 27]
        expected_data = np.array([[[[7, 11], [23, 27]]]], dtype=np.float32)
        self.assertTrue(np.allclose(output.data, expected_data))
        self.assertEqual(output.shape, (1, 1, 2, 2))
        self.assertTrue(output.requires_grad)

    def test_conv2dTranspose_basic(self):
        """Test basic 2D transposed convolution with known input and kernel."""
        # Input: (1, 1, 2, 2) - 1 batch, 1 channel, 2x2 spatial dimensions
        input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        # Kernel: (1, 1, 2, 2) - all ones for simplicity
        kernel_data = np.array([[[[1, 1], [1, 1]]]], dtype=np.float32)
        x = Tensor(input_data, requires_grad=True)
        kernel = Tensor(kernel_data, requires_grad=True)
        output = x.conv2dTranspose(kernel, padding=0, stride=1)
        # Expected output: (1, 1, 3, 3)
        # Computed by placing kernel at each input position and summing overlaps:
        # [1, 1+2, 2]
        # [1+3, 1+2+3+4, 2+4]
        # [3, 3+4, 4]
        expected_data = np.array(
            [[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]], dtype=np.float32
        )
        self.assertTrue(np.allclose(output.data, expected_data))
        self.assertEqual(output.shape, (1, 1, 3, 3))
        self.assertTrue(output.requires_grad)

    def test_conv2dTranspose_stride(self):
        """Test 2D transposed convolution with stride > 1."""
        # Input: (1, 1, 2, 2)
        input_data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        # Kernel: (1, 1, 2, 2) - all ones
        kernel_data = np.array([[[[1, 1], [1, 1]]]], dtype=np.float32)
        x = Tensor(input_data, requires_grad=True)
        kernel = Tensor(kernel_data, requires_grad=True)
        output = x.conv2dTranspose(kernel, padding=0, stride=2)
        # Expected output: (1, 1, 4, 4)
        # With stride=2, each input spreads to a 2x2 block, non-overlapping:
        # [1,1,2,2]
        # [1,1,2,2]
        # [3,3,4,4]
        # [3,3,4,4]
        expected_data = np.array(
            [[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(output.data, expected_data))
        self.assertEqual(output.shape, (1, 1, 4, 4))
        self.assertTrue(output.requires_grad)

    def test_relu(self):
        """Test ReLU forward pass with positive, negative, and zero values."""
        data = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.relu()

        expected = np.array([0.0, 0.0, 0.0, 1.0, 3.0], dtype=np.float32)
        self.assertTrue(np.array_equal(out.data, expected))
        self.assertTrue(out.requires_grad)

    def test_log(self):
        """Test natural log forward pass."""
        data = np.array([1.0, np.e, np.e**2], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.log()

        expected = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.assertTrue(np.allclose(out.data, expected, atol=1e-6))
        self.assertTrue(out.requires_grad)

    def test_exp(self):
        """Test exponential forward pass."""
        data = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.exp()

        expected = np.array([1.0, np.e, np.e**2], dtype=np.float32)
        self.assertTrue(np.allclose(out.data, expected, atol=1e-6))
        self.assertTrue(out.requires_grad)

    def test_abs(self):
        """Test absolute value forward pass."""
        data = np.array([-3.0, 0.0, 4.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.abs()

        expected = np.array([3.0, 0.0, 4.0], dtype=np.float32)
        self.assertTrue(np.array_equal(out.data, expected))
        self.assertTrue(out.requires_grad)

    def test_sum(self):
        """Test tensor summation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.sum()

        self.assertEqual(out.data, 10.0)
        self.assertTrue(out.requires_grad)

    def test_no_grad_path(self):
        """Test operations when requires_grad=False."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = Tensor(data, requires_grad=False)

        # All operations should produce non-requires-grad tensors
        self.assertFalse(t.relu().requires_grad)
        self.assertFalse(t.log().requires_grad)
        self.assertFalse(t.exp().requires_grad)
        self.assertFalse(t.abs().requires_grad)
        self.assertFalse(t.sum().requires_grad)

        # Should not accumulate gradients
        out = t.exp().sum()
        with self.assertRaises(RuntimeError):
            out.backward()

    def test_log_stability(self):
        data = np.array([0.0, 1e-8, 1e-16], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.log().sum().backward()

        # The expected gradient is now 1 / (x + 1e-8)
        expected_grad = 1 / (data + 1e-8)
        self.assertTrue(np.allclose(t.grad, expected_grad, atol=1e-6))

    def test_exp_overflow(self):
        """Test exp with large values doesn't break gradient."""
        data = np.array([100.0], dtype=np.float32)
        t = Tensor(data, requires_grad=True)
        out = t.exp()

        # Should get inf in forward pass but still valid gradient
        self.assertTrue(np.isinf(out.data[0]))

        # Backward should propagate inf * grad
        out.backward()
        self.assertTrue(np.isinf(t.grad[0]))


if __name__ == "__main__":
    unittest.main()
