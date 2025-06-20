import numpy as np

from .autograd import Autograd
from .tensor import Tensor


class Ops:
    @staticmethod
    def add(a: Tensor, b: Tensor) -> Tensor:
        data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad

        ctx, _grad_fn = None, None

        if requires_grad:
            ctx = {"inputs": (a, b), "shape": data.shape}
            _grad_fn = Autograd.add_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def sub(a: Tensor, b: Tensor) -> Tensor:
        data = a.data - b.data
        requires_grad = a.requires_grad or b.requires_grad

        ctx, _grad_fn = None, None

        if requires_grad:
            ctx = {"inputs": (a, b), "shape": data.shape}
            _grad_fn = Autograd.sub_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def mul(a: Tensor, scalar: np.float32) -> Tensor:
        data = a.data * scalar
        requires_grad = a.requires_grad

        ctx, _grad_fn = None, None

        if requires_grad:
            ctx = {"inputs": (a, scalar), "shape": data.shape}
            _grad_fn = Autograd.mul_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def elementwisemul(a: Tensor, b: Tensor) -> Tensor:
        data = np.multiply(a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad

        ctx, _grad_fn = None, None

        if requires_grad:
            ctx = {"inputs": (a, b), "shape": data.shape}
            _grad_fn = Autograd.elementwisemul_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def matmul(a: Tensor, b: Tensor) -> Tensor:
        data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad

        ctx, _grad_fn = None, None

        if requires_grad:
            ctx = {
                "inputs": (a, b),
                "shape": data.shape,
            }
            _grad_fn = Autograd.matmul_backward

        return Tensor(data, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def conv2d(
        t: Tensor,
        kernel: Tensor,
        padding: tuple = (0, 0),
        stride: tuple = (1, 1),
    ) -> Tensor:
        N, C_in, H, W = t.shape
        C_out, C_in_k, Kh, Kw = kernel.shape
        assert C_in == C_in_k, "Input channels must match kernel channels"

        if isinstance(stride, tuple):
            Sh, Sw = stride
        else:
            Sh, Sw = (stride, stride)
        if isinstance(padding, tuple):
            Ph, Pw = padding
        else:
            Ph, Pw = (padding, padding)

        H_out = (H + 2 * Ph - Kh) // Sh + 1
        W_out = (W + 2 * Pw - Kw) // Sw + 1
        assert H_out > 0 and W_out > 0, "Invalid output dimensions"

        if Ph > 0 or Pw > 0:
            t_padded = np.pad(
                t.data, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)), mode="constant"
            )
        else:
            t_padded = t.data

        output = np.zeros((N, C_out, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                patch = t_padded[:, :, i * Sh : i * Sh + Kh, j * Sw : j * Sw + Kw]
                for c_out in range(C_out):
                    filter = kernel.data[c_out]
                    output[:, c_out, i, j] = np.sum(patch * filter, axis=(1, 2, 3))

        requires_grad = t.requires_grad or kernel.requires_grad
        ctx = {
            "inputs": (t, kernel),
            "shape": output.shape,
            "padding": padding,
            "stride": stride,
        }
        _grad_fn = Autograd.conv2d_backward

        return Tensor(output, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))

    @staticmethod
    def conv2dTranspose(
        t: Tensor, kernel: Tensor, padding: tuple = (0, 0), stride: tuple = (1, 1)
    ) -> Tensor:
        N, C_in, H, W = t.shape
        C_out, _, Kh, Kw = kernel.shape

        # Handle stride and padding as tuples or scalars
        if isinstance(stride, tuple):
            Sh, Sw = stride
        else:
            Sh, Sw = (stride, stride)
        if isinstance(padding, tuple):
            Ph, Pw = padding
        else:
            Ph, Pw = (padding, padding)

        # Calculate output dimensions
        Oh = (H - 1) * Sh + Kh - 2 * Ph
        Ow = (W - 1) * Sw + Kw - 2 * Pw

        # Validate output dimensions
        if Oh <= 0 or Ow <= 0:
            raise ValueError(
                f"Invalid output dimensions: Oh={Oh}, Ow={Ow}. Adjust input size, stride, padding, or kernel size."
            )

        # Initialize output tensor
        output = np.zeros((N, C_out, Oh, Ow))

        # Perform transposed convolution
        for i in range(H):
            for j in range(W):
                for c_out in range(C_out):
                    for c_in in range(C_in):
                        input_val = t.data[:, c_in, i, j]
                        kernel_patch = kernel.data[c_out, c_in, :, :]

                        for kh in range(Kh):
                            for kw in range(Kw):
                                out_i = i * Sh + kh - Ph
                                out_j = (
                                    j * Sw + kw - Pw
                                )  # Corrected from i * Sw to j * Sw
                                if 0 <= out_i < Oh and 0 <= out_j < Ow:
                                    output[:, c_out, out_i, out_j] += (
                                        input_val * kernel_patch[kh, kw]
                                    )

        # Determine if gradients are required
        requires_grad = t.requires_grad or kernel.requires_grad
        _grad_fn, ctx = None, None

        # Set up autograd context if gradients are needed
        if requires_grad:
            ctx = {
                "inputs": (t, kernel),
                "shape": output.shape,
                "padding": padding,
                "stride": stride,
            }
            _grad_fn = Autograd.conv2dTranspose_backward

        # Return the output tensor
        return Tensor(output, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))
