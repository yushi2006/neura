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
        x_tensor, W_tensor = inputs[0], inputs[1]

        x = x_tensor.data
        W = W_tensor.data

        Ph, Pw = ctx.get("padding", (0, 0))
        Sh, Sw = ctx.get("stride", (1, 1))

        N, C_in, H, W1 = x.shape
        C_out, _, Kh, Kw = W.shape
        _, _, H_out, W_out = grad_output.shape

        if Ph > 0 or Pw > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)), mode='constant')
        else:
            x_padded = x
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(W)

        for n in range(N):
            for cout in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        grad_val = grad_output[n, cout, i, j]
                        h_start = i * Sh
                        w_start = j * Sw
                        patch = x_padded[n, :, h_start:h_start+Kh, w_start:w_start+Kw]
                        if W_tensor.requires_grad:
                            dW[cout] += patch * grad_val
                        if x_tensor.requires_grad:
                            dx_padded[n, :, h_start:h_start+Kh, w_start:w_start+Kw] += W[cout] * grad_val

        if Ph > 0 or Pw > 0:
            dx = dx_padded[:, :, Ph:Ph+H, Pw:Pw+W1]
        else:
            dx = dx_padded

        return dx, dW

