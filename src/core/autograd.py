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
    def matmul_backward(grad, ctx):
        a, b = ctx["inputs"]

        da = grad @ b.T
        db = a.T @ grad

        return da, db
