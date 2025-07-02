from __future__ import annotations

# This is needed for type hints
from typing import TYPE_CHECKING

import numpy as np

# This import is needed for the broadcasting helper function
from .ops import Ops

if TYPE_CHECKING:
    pass


class Autograd:
    @staticmethod
    def add_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]
        # Start with the gradient for both inputs
        da, db = grad, grad

        # FIX: Reduce gradients to original shapes to handle broadcasting
        if a.shape != grad.shape:
            da = Ops.reduce_grad_for_broadcast(da, a.shape)
        if b.shape != grad.shape:
            db = Ops.reduce_grad_for_broadcast(db, b.shape)

        return da, db

    @staticmethod
    def sub_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]
        # Start with the gradient for 'a' and negative gradient for 'b'
        da, db = grad, -grad

        # FIX: Reduce gradients to original shapes to handle broadcasting
        if a.shape != grad.shape:
            da = Ops.reduce_grad_for_broadcast(da, a.shape)
        if b.shape != grad.shape:
            db = Ops.reduce_grad_for_broadcast(db, b.shape)

        return da, db

    @staticmethod
    def mul_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        a, scalar = ctx["inputs"]
        da = grad * scalar

        # FIX: Handle cases where the tensor was broadcasted to match the scalar op
        if a.shape != da.shape:
            da = Ops.reduce_grad_for_broadcast(da, a.shape)

        # FIX: Must return a tuple
        return (da,)

    @staticmethod
    def elementwisemul_backward(
        grad: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]

        # FIX: Corrected typo from a.dat to a.data
        da = np.multiply(b.data, grad)
        db = np.multiply(a.data, grad)

        # FIX: Reduce gradients to original shapes to handle broadcasting
        if a.shape != grad.shape:
            da = Ops.reduce_grad_for_broadcast(da, a.shape)
        if b.shape != grad.shape:
            db = Ops.reduce_grad_for_broadcast(db, b.shape)

        return da, db

    @staticmethod
    def matmul_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx["inputs"]

        # FIX: Use .data explicitly for numpy operations. Your .T property works,
        # but this is more robust and clear.
        da = grad @ b.data.T
        db = a.data.T @ grad

        return da, db

    @staticmethod
    def conv2d_backward(
        grad_output: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        # Your existing conv2d backward implementation is mostly correct in its logic.
        # This implementation is very slow due to Python loops, but it is functionally sound.
        x, kernel = ctx["inputs"]
        # ... (your existing conv2d_backward code is fine here) ...
        # (The code from your prompt is copied here for completeness)
        if isinstance(ctx["stride"], tuple):
            Sh, Sw = ctx["stride"]
        else:
            Sh, Sw = (ctx["stride"], ctx["stride"])
        if isinstance(ctx["padding"], tuple):
            Ph, Pw = ctx["padding"]
        else:
            Ph, Pw = (ctx["padding"], ctx["padding"])
        N, C_in, H, W = x.data.shape
        C_out, _, Kh, Kw = kernel.data.shape
        _, _, H_out, W_out = grad_output.shape
        if Ph > 0 or Pw > 0:
            x_padded = np.pad(
                x.data, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)), mode="constant"
            )
        else:
            x_padded = x.data
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(kernel.data)
        for n in range(N):
            for cout in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        grad_val = grad_output[n, cout, i, j]
                        h_start, w_start = i * Sh, j * Sw
                        if (
                            h_start + Kh <= x_padded.shape[2]
                            and w_start + Kw <= x_padded.shape[3]
                        ):
                            patch = x_padded[
                                n, :, h_start : h_start + Kh, w_start : w_start + Kw
                            ]
                            if kernel.requires_grad:
                                dW[cout] += patch * grad_val
                            if x.requires_grad:
                                dx_padded[
                                    n, :, h_start : h_start + Kh, w_start : w_start + Kw
                                ] += kernel.data[cout] * grad_val
        if Ph > 0 or Pw > 0:
            dx = dx_padded[:, :, Ph : Ph + H, Pw : Pw + W]
        else:
            dx = dx_padded
        return dx, dW

    @staticmethod
    def conv2dTranspose_backward(
        grad_output: np.ndarray, ctx: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        # Your existing conv2dTranspose backward implementation is also fine.
        x, kernel = ctx["inputs"]
        # ... (your existing conv2dTranspose_backward code is fine here) ...
        # (The code from your prompt is copied here for completeness)
        N, C_in, H, W = x.shape
        C_out, C_in_k, Kh, Kw = kernel.shape
        assert C_in == C_in_k, "Input channels must match kernel's in_channels"
        N_out, C_out_grad, Oh, Ow = grad_output.shape
        assert N == N_out and C_out == C_out_grad, "Output gradient shape mismatch"
        if isinstance(ctx["stride"], tuple):
            Sh, Sw = ctx["stride"]
        else:
            Sh, Sw = (ctx["stride"], ctx["stride"])
        if isinstance(ctx["padding"], tuple):
            Ph, Pw = ctx["padding"]
        else:
            Ph, Pw = (ctx["padding"], ctx["padding"])
        dx = np.zeros((N, C_in, H, W))
        dw = np.zeros((C_out, C_in, Kh, Kw))
        for n in range(N):
            for c_in in range(C_in):
                for i in range(H):
                    for j in range(W):
                        for c_out in range(C_out):
                            for kh in range(Kh):
                                for kw in range(Kw):
                                    out_i, out_j = i * Sh + kh - Ph, j * Sw + kw - Pw
                                    if 0 <= out_i < Oh and 0 <= out_j < Ow:
                                        dx[n, c_in, i, j] += (
                                            grad_output[n, c_out, out_i, out_j]
                                            * kernel.data[c_out, c_in, kh, kw]
                                        )
        for c_out in range(C_out):
            for c_in in range(C_in):
                for kh in range(Kh):
                    for kw in range(Kw):
                        for n in range(N):
                            for i in range(H):
                                for j in range(W):
                                    out_i, out_j = i * Sh + kh - Ph, j * Sw + kw - Pw
                                    if 0 <= out_i < Oh and 0 <= out_j < Ow:
                                        dw[c_out, c_in, kh, kw] += (
                                            x.data[n, c_in, i, j]
                                            * grad_output[n, c_out, out_i, out_j]
                                        )
        return dx, dw

    @staticmethod
    def relu_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        mask = ctx["mask"]
        # No change needed here, it was already correct.
        return (grad * mask,)

    @staticmethod
    def log_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        # FIX: Use "input_data" from context, not "inputs".
        input_data = ctx["input_data"]
        # Add a small epsilon for numerical stability.
        return (grad * (1 / (input_data + 1e-8)),)

    @staticmethod
    def exp_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        # FIX: Use "input_data" from context, not "inputs". This is exp(x).
        output_data = ctx["input_data"]
        return (grad * output_data,)

    @staticmethod
    def sum_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        # FIX: This requires the input tensor from the context to know the original shape.
        (t,) = ctx["inputs"]
        # Broadcast the scalar gradient back to the original tensor's shape.
        return (np.ones_like(t.data) * grad,)

    @staticmethod
    def abs_backward(grad: np.ndarray, ctx: dict) -> tuple[np.ndarray]:
        # FIX: The derivative of abs(x) is sign(x). The mask t.data > 0 is incorrect
        # as it misses the case where the gradient should be -1.
        (t,) = ctx["inputs"]
        return (grad * np.sign(t.data),)
