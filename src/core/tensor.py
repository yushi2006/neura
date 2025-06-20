from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .init_strategies import InitStrategy, OneInit, RandnInit, ZeroInit


class Tensor:
    def __init__(
        self,
        data: Union[List, np.ndarray],
        requires_grad: bool = True,
        dtype: type = np.float32,
        _ctx: Optional[tuple] = None,
    ):
        if not isinstance(data, np.ndarray):
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = data
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._ctx = _ctx
        self.T = self.data.T

    @classmethod
    def from_strategy(
        cls,
        shape: tuple,
        strategy: InitStrategy,
        requires_grad: bool = True,
        dtype: type = np.float32,
    ) -> Tensor:
        data = strategy.init(shape, dtype)
        return cls(data, requires_grad, dtype=dtype)

    @classmethod
    def ones(cls, shape: tuple, **kwargs) -> Tensor:
        return cls.from_strategy(shape, OneInit(), **kwargs)

    @classmethod
    def zeros(cls, shape: tuple, **kwargs) -> Tensor:
        return cls.from_strategy(shape, ZeroInit(), **kwargs)

    @classmethod
    def randn(cls, shape: tuple, **kwargs) -> Tensor:
        return cls.from_strategy(shape, RandnInit(), **kwargs)

    def __add__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        return Ops.add(self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        return Ops.sub(self, other)

    def __mul__(self, other: Union[np.float32, Tensor]) -> Tensor:
        from .ops import Ops

        if isinstance(other, Tensor):
            return Ops.elementwisemul(self, b=other)
        else:
            return Ops.mul(self, scalar=other)

    def __matmul__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        return Ops.matmul(self, other)

    def __iadd__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        self = Ops.add(self, other)

        return self

    def __isub__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        self = Ops.sub(self, other)

        return self

    def __imul__(self, scalar: np.float32) -> Tensor:
        from .ops import Ops

        self = Ops.mul(self, scalar=scalar)

        return self

    def __imatmul__(self, other: Tensor) -> Tensor:
        from .ops import Ops

        self = Ops.matmul(self, other)

        return self

    def conv2d(self, kernel: Tensor, **kwargs) -> Tensor:
        from .ops import Ops

        output = Ops.conv2d(self, kernel, **kwargs)

        return output

    def conv2dTranspose(self, kernel: Tensor, **kwargs) -> Tensor:
        from .ops import Ops

        output = Ops.conv2dTranspose(self, kernel, **kwargs)

        return output

    def __getitem__(self, idx: Union[int, slice]) -> Tensor:
        result = self.data[idx]

        return Tensor(result, requires_grad=self.requires_grad, _ctx=self._ctx)

    def __len__(self) -> int:
        return self.data.size

    def view(self, *args: int) -> Tensor:
        self.data = self.data.reshape(*args)

        return self

    def squeeze(self, dim: int) -> Tensor:
        self.data = self.data.squeeze(axis=dim)

        return self

    def unsqueeze(self, dim: int) -> Tensor:
        self.data = np.expand_dims(self.data, axis=dim)

        return self

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Called backward on non-require-grad tensor.")

        if grad is None:
            if self.data.size == 1:
                raise RuntimeError(
                    "grad can be implicity created only for non-scalar outputs."
                )
            grad = np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad

        if self._ctx is None:
            return

        backward_fn, arg_ctx = self._ctx
        inputs = arg_ctx["inputs"]

        input_grads = backward_fn(grad, arg_ctx)

        for t, g in zip(inputs, input_grads):
            if isinstance(t, Tensor):
                if t.requires_grad:
                    t.backward(g)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        if self._ctx is not None:
            _, ctx = self._ctx
            for prev in ctx["inputs"]:
                prev.zero_grad()

    def __repr__(self) -> str:
        return f"{self.data}, dtype={self.dtype}"
