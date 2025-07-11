import unittest

import neura
import numpy as np
from neura.nn import Conv2d, Conv2dTranspose  # Replace with actual import path


class TestConv2d(unittest.TestCase):
    def test_initialization(self):
        in_channels = 3
        out_channels = 5
        kernel_size = 3
        conv = Conv2d(in_channels, out_channels, kernel_size, has_bias=True)

        # Check weight shape and properties
        self.assertEqual(
            conv.W.shape, (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.assertIsInstance(conv.W, neura.Tensor)
        self.assertTrue(conv.W.requires_grad)

        # Check bias shape and properties
        self.assertEqual(conv.b.shape, (out_channels, 1, 1))
        self.assertIsInstance(conv.b, neura.Tensor)
        self.assertTrue(
            conv.b.requires_grad
        )  # Assuming bias doesn't require grad by default

        # Check without bias
        conv_no_bias = Conv2d(in_channels, out_channels, kernel_size, has_bias=False)
        self.assertIsNone(conv_no_bias.b)

    def test_forward_shape(self):
        conv = Conv2d(1, 1, 3, padding=0, stride=1, has_bias=False)
        x = neura.Tensor(np.ones((1, 1, 5, 5), dtype=np.float32))
        output = conv.forward(x)
        self.assertEqual(
            output.shape, (1, 1, 3, 3)
        )  # 5x5 input with 3x3 kernel -> 3x3 output

    def test_forward_with_bias(self):
        conv = Conv2d(1, 1, 2, padding=0, stride=1, has_bias=True)
        conv.W.data = np.ones((1, 1, 2, 2), dtype=np.float32)
        conv.b.data = np.ones((1, 1, 1), dtype=np.float32)
        x = neura.Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))
        output = conv.forward(x)
        # Each output element: 4 (from 2x2 kernel of ones) + 1 (bias) = 5
        np.testing.assert_array_almost_equal(output.data, np.full((1, 1, 2, 2), 5.0))

    def test_channel_mismatch(self):
        conv = Conv2d(2, 1, 3)
        x = neura.Tensor(np.ones((1, 1, 5, 5), dtype=np.float32))  # Wrong in_channels
        with self.assertRaises(ValueError):
            conv.forward(x)

    def test_padding_stride(self):
        conv = Conv2d(1, 1, 3, padding=1, stride=2, has_bias=False)
        conv.W.data = np.ones((1, 1, 3, 3), dtype=np.float32)
        x = neura.Tensor(np.ones((1, 1, 5, 5), dtype=np.float32))
        output = conv.forward(x)
        self.assertEqual(output.shape, (1, 1, 3, 3))  # With padding=1, stride=2


class TestConv2dTranspose(unittest.TestCase):
    def test_initialization(self):
        in_channels = 2
        out_channels = 4
        kernel_size = 4
        conv_trans = Conv2dTranspose(
            in_channels, out_channels, kernel_size, has_bias=True
        )

        # Check weight shape and properties
        self.assertEqual(
            conv_trans.W.shape, (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.assertIsInstance(conv_trans.W, neura.Tensor)
        self.assertTrue(conv_trans.W.requires_grad)

        # Check bias shape and properties
        self.assertEqual(conv_trans.b.shape, (out_channels, 1, 1))
        self.assertIsInstance(conv_trans.b, neura.Tensor)
        self.assertTrue(conv_trans.b.requires_grad)

        # Check without bias
        conv_trans_no_bias = Conv2dTranspose(
            in_channels, out_channels, kernel_size, has_bias=False
        )
        self.assertIsNone(conv_trans_no_bias.b)

    def test_forward_shape(self):
        conv_trans = Conv2dTranspose(1, 1, 2, padding=0, stride=1, has_bias=False)
        x = neura.Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))
        output = conv_trans.forward(x)
        self.assertEqual(output.shape, (1, 1, 4, 4))  # Typical transposed conv upsizing

    def test_forward_with_bias(self):
        conv_trans = Conv2dTranspose(1, 1, 2, padding=0, stride=1, has_bias=True)
        conv_trans.W.data = np.ones((1, 1, 2, 2), dtype=np.float32)
        conv_trans.b.data = np.ones((1, 1, 1), dtype=np.float32)
        x = neura.Tensor(np.ones((1, 1, 2, 2), dtype=np.float32))
        output = conv_trans.forward(x)
        # Assuming transposed conv behavior, check if bias is applied
        self.assertTrue(np.all(output.data >= 1.0))  # Values should include bias

    def test_channel_mismatch(self):
        conv_trans = Conv2dTranspose(3, 1, 3)
        x = neura.Tensor(np.ones((1, 2, 5, 5), dtype=np.float32))  # Wrong in_channels
        with self.assertRaises(ValueError):
            conv_trans.forward(x)


if __name__ == "__main__":
    unittest.main()
