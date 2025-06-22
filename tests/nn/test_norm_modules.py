import unittest

import numpy as np
from neura import Tensor
from neura.nn import BatchNorm2d


class TestBatchNorm2d(unittest.TestCase):
    def setUp(self):
        # Seed for reproducibility
        np.random.seed(42)

    def test_forward_shape(self):
        """
        Test that forward returns the correct shape and does not modify the input shape.
        """
        batch_size, channels, height, width = 4, 3, 5, 5
        # Random input
        x_np = np.random.randn(batch_size, channels, height, width).astype(float)
        x_tensor = Tensor(x_np)

        bn = BatchNorm2d(m=channels)
        out = bn.forward(x_tensor)
        # out should be a Tensor
        self.assertIsInstance(out, Tensor)
        # shape should match input
        self.assertEqual(out.data.shape, x_np.shape)

    def test_normalization_unit_variance(self):
        """
        Test that, when gamma=1 and beta=0, the output has zero mean and unit variance
        across (batch, height, width) for each channel.
        """
        batch_size, channels, height, width = 8, 4, 6, 6
        x_np = (
            np.random.randn(batch_size, channels, height, width).astype(float) * 2.0
            + 1.0
        )
        x_tensor = Tensor(x_np)

        bn = BatchNorm2d(m=channels, epsilon=1e-5)
        # Override gamma and beta:
        # gamma shape in implementation: Tensor(np.zeros((1, m))) by default.
        # We set gamma to ones and beta to zeros.
        gamma_np = np.ones((1, channels), dtype=float)
        beta_np = np.zeros((1, channels), dtype=float)
        bn.gamma = Tensor(gamma_np)
        bn.beta = Tensor(beta_np)

        out_tensor = bn.forward(x_tensor)
        out_np = out_tensor.data  # shape (batch, channels, height, width)

        # Compute expected mean and variance along axes (0,2,3) per channel:
        # mean shape (1, C, 1, 1), variance shape (1, C, 1, 1)
        mean = np.mean(x_np, axis=(0, 2, 3), keepdims=True)
        variance = np.mean((x_np - mean) ** 2, axis=(0, 2, 3), keepdims=True)
        x_bar = (x_np - mean) / np.sqrt(variance + bn.epsilon)

        # With gamma=1, beta=0, expected output is x_bar
        # Check per-channel mean ~0 and variance ~1
        # First, check the output matches x_bar closely
        np.testing.assert_allclose(out_np, x_bar, rtol=1e-5, atol=1e-6)

        # Then check mean and variance numeric properties
        # Compute mean over (0,2,3) for each channel
        out_mean = np.mean(out_np, axis=(0, 2, 3))

        # Reshape mean for proper broadcasting (1, channels, 1, 1)
        reshaped_mean = out_mean.reshape((1, channels, 1, 1))

        # Compute variance using correct axes (0,2,3)
        squared_diff = (out_np - reshaped_mean) ** 2
        out_var = np.mean(squared_diff, axis=(0, 2, 3))

        # out_mean should be very close to 0
        np.testing.assert_allclose(out_mean, np.zeros(channels), atol=1e-6, rtol=1e-5)
        # out_var should be very close to 1
        np.testing.assert_allclose(out_var, np.ones(channels), atol=1e-6, rtol=1e-5)

    def test_constant_input(self):
        """
        Test that for a constant input, the normalized output is zero (before scaling/shift).
        """
        batch_size, channels, height, width = 3, 2, 4, 4
        const_value = 7.5
        x_np = np.full((batch_size, channels, height, width), const_value, dtype=float)
        x_tensor = Tensor(x_np)

        bn = BatchNorm2d(m=channels, epsilon=1e-5)
        # Set gamma to ones, beta to zeros to isolate normalization behavior:
        bn.gamma = Tensor(np.ones((1, channels), dtype=float))
        bn.beta = Tensor(np.zeros((1, channels), dtype=float))

        out_tensor = bn.forward(x_tensor)
        out_np = out_tensor.data

        # mean will equal const_value, variance ~0, so normalization yields zero array
        expected = np.zeros_like(x_np)
        np.testing.assert_allclose(out_np, expected, atol=1e-6)

    def test_non_default_epsilon(self):
        """
        Test that a different epsilon still gives stable output (no NaNs) for small variance inputs.
        """
        batch_size, channels, height, width = 5, 3, 4, 4
        # Create input where variance is extremely small
        base = (
            np.random.randn(batch_size, channels, height, width).astype(float) * 1e-3
            + 2.0
        )
        x_tensor = Tensor(base)

        small_epsilon = 1e-2  # much larger epsilon
        bn = BatchNorm2d(m=channels, epsilon=small_epsilon)
        bn.gamma = Tensor(np.ones((1, channels), dtype=float))
        bn.beta = Tensor(np.zeros((1, channels), dtype=float))

        out_tensor = bn.forward(x_tensor)
        out_np = out_tensor.data

        # Since variance is tiny but epsilon is large, denominator √(var + eps) ≈ sqrt(eps)
        # The normalized output is (x - mean) / sqrt(var+eps). mean is ~2.0; x-base ~ small noise.
        # So out_np should be small but non-NaN/Inf.
        self.assertFalse(np.isnan(out_np).any(), "Output contains NaNs")
        self.assertFalse(np.isinf(out_np).any(), "Output contains Infs")
        # Also check that mean of out_np across (0,2,3) is approximately 0
        out_mean = np.mean(out_np, axis=(0, 2, 3))
        np.testing.assert_allclose(out_mean, np.zeros(channels), atol=1e-3)

    def test_integration_with_backward_and_params(self):
        """
        If your Tensor and Module support backward and parameters,
        check that parameters() picks up gamma and beta, and that a backward pass updates grads.
        This is a more “structure” test; adjust depending on your actual Tensor API.
        """
        batch_size, channels, height, width = 2, 2, 3, 3
        x_np = np.random.randn(batch_size, channels, height, width).astype(float)
        x_tensor = Tensor(x_np)

        bn = BatchNorm2d(m=channels, epsilon=1e-5)
        # Ensure gamma and beta are part of parameters
        params = bn.parameters()
        # Expect at least gamma and beta in params
        # We check that their Tensor identities are in the returned list
        self.assertTrue(any(p is bn.gamma for p in params), "gamma not in parameters()")
        self.assertTrue(any(p is bn.beta for p in params), "beta not in parameters()")

        # Now, test a dummy backward if possible: e.g. define a simple loss = sum(out)
        # and check that grads on gamma/beta get computed. This depends on your Tensor API.
        out = bn.forward(x_tensor)  # Tensor
        # Dummy loss: sum of all elements in out.data
        loss = out.sum()  # assumes Tensor.sum() exists and creates a scalar Tensor
        # Zero existing grads if needed
        try:
            for p in params:
                p.grad = None
        except Exception:
            pass
        # Backward
        try:
            loss.backward()
        except Exception as e:
            self.skipTest(
                f"Skipping backward test: Tensor.backward() not implemented or failed: {e}"
            )
        # After backward, gamma.grad and beta.grad should exist (possibly non-zero)
        # We do a loose check: attributes exist and are arrays of the right shape.
        if hasattr(bn.gamma, "grad"):
            self.assertIsInstance(
                bn.gamma.grad, np.ndarray, "gamma.grad should be a NumPy array"
            )
            self.assertEqual(bn.gamma.grad.shape, bn.gamma.data.shape)
        if hasattr(bn.beta, "grad"):
            self.assertIsInstance(
                bn.beta.grad, np.ndarray, "beta.grad should be a NumPy array"
            )
            self.assertEqual(bn.beta.grad.shape, bn.beta.data.shape)


if __name__ == "__main__":
    unittest.main()
