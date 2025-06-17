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
