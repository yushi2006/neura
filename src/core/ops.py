import numpy as np
from .tensor import Tensor
from .autograd import Autograd


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
    def mul(a: Tensor, scalar: np.float16) -> Tensor:
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
    def conv2d(t: Tensor, kernel: Tensor, padding: tuple = (0, 0), stride: tuple = (1, 1)) -> Tensor:
        N, C_in, H, W = t.shape
        C_out, C_in_k, Kh, Kw = kernel.shape
        assert C_in == C_in_k, "Input channels must match kernel channels"
        Ph, Pw = padding
        Sh, Sw = stride

        H_out = (H + 2 * Ph - Kh) // Sh + 1
        W_out = (W + 2 * Pw - Kw) // Sw + 1
        assert H_out > 0 and W_out > 0, "Invalid output dimensions"

        if Ph > 0 or Pw > 0:
            t_padded = np.pad(t.data, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)), mode='constant')
        else:
            t_padded = t.data

        output = np.zeros((N, C_out, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                patch = t_padded[:, :, i * Sh:i * Sh + Kh, j * Sw:j * Sw + Kw]
                for c_out in range(C_out):
                    filter = kernel.data[c_out]
                    output[:, c_out, i, j] = np.sum(patch * filter, axis=(1, 2, 3))


        requires_grad = t.requires_grad or kernel.requires_grad
        ctx = {
            "inputs": (t, kernel),
            "shape": output.shape,
            "padding": padding,
            "stride": stride
        }
        _grad_fn = Autograd.conv2d_backward

        return Tensor(output, requires_grad=requires_grad, _ctx=(_grad_fn, ctx))
