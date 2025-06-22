from __future__ import annotations

import numpy as np


class Autograd:
    @staticmethod
    def add_backward(grad, ctx):
        a, b = ctx["inputs"]
        da = grad * np.ones_like(a.data)
        db = grad * np.ones_like(b.data)

        return da, db

    @staticmethod
    def sub_backward(grad, ctx):
        a, b = ctx["inputs"]
        da = grad * np.ones_like(a.data)
        db = grad * -np.ones_like(b.data)

        return da, db

    @staticmethod
    def mul_backward(grad, ctx):
        _, scalar = ctx["inputs"]

        da = grad * scalar

        return da

    @staticmethod
    def elementwisemul_backward(grad, ctx):
        a, b = ctx["inputs"]

        da = np.multiply(b.data, grad)
        db = np.multiply(a.dat, grad)

        return da, db

    @staticmethod
    def matmul_backward(grad, ctx):
        a, b = ctx["inputs"]

        da = grad @ b.T
        db = a.T @ grad

        return da, db

    @staticmethod
    def conv2d_backward(grad_output, ctx):
        inputs = ctx["inputs"]
        x, kernel = inputs

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
                        h_start = i * Sh
                        w_start = j * Sw
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
    def conv2dTranspose_backward(grad_output, ctx):
        x, kernel = ctx["inputs"]
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
                                    out_i = i * Sh + kh - Ph
                                    out_j = j * Sw + kw - Pw
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
                                    out_i = i * Sh + kh - Ph
                                    out_j = j * Sw + kw - Pw
                                    if 0 <= out_i < Oh and 0 <= out_j < Ow:
                                        dw[c_out, c_in, kh, kw] += (
                                            x.data[n, c_in, i, j]
                                            * grad_output[n, c_out, out_i, out_j]
                                        )

        return dx, dw
