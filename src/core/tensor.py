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

    def relu(self) -> Tensor:
        from .ops import Ops

        output = Ops.relu(self)
        return output

    def abs(self) -> Tensor:
        from .ops import Ops

        output = Ops.abs(self)
        return output

    def log(self) -> Tensor:
        from .ops import Ops

        output = Ops.log(self)
        return output

    def exp(self) -> Tensor:
        from .ops import Ops

        output = Ops.exp(self)
        return output

    def sum(self) -> Tensor:
        from .ops import Ops

        output = Ops.sum(self)
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

    def broadcast_to(self, shape: tuple) -> "Tensor":
        from .ops import Ops

        try:
            out_data = np.broadcast_to(self.data, shape)
        except Exception as e:
            raise ValueError(
                f"Cannot broadcast Tensor of shape {self.shape} to target shape {shape}: {e}"
            )

        def backward_fn(upstream_grad: np.ndarray, ctx):
            orig_shape = ctx["orig_shape"]
            grad_self = Ops.reduce_grad_for_broadcast(upstream_grad, orig_shape)
            return [grad_self]

        ctx = {"inputs": (self,), "orig_shape": self.shape}
        requires_grad = self.requires_grad
        return Tensor(out_data, requires_grad=requires_grad, _ctx=(backward_fn, ctx))

    @staticmethod
    def build_topo(tensor: "Tensor") -> list["Tensor"]:
        """
        Performs a topological sort of the graph ending at this tensor.
        """
        topo = []
        visited = set()

        def _visit(t):
            if t not in visited:
                visited.add(t)
                if t._ctx:
                    for parent in t._ctx[1]["inputs"]:
                        if isinstance(parent, Tensor):
                            _visit(parent)
                topo.append(t)

        _visit(tensor)
        return topo

    def backward(self, grad: Optional[np.ndarray] = None):
        if not self.requires_grad:
            raise RuntimeError(
                "Called backward on a tensor that does not require gradients."
            )

        # --- THE FIX IS HERE ---
        if grad is None:
            # Instead of raising an error for non-scalars, we create a
            # gradient of ones. This is a common convention for testing.
            grad = np.ones_like(self.data, dtype=self.dtype)

        # The `grad` attribute accumulates gradients.
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        # --- The rest of your backward logic is correct ---
        topo_sorted_graph = self.build_topo(self)

        for t in reversed(topo_sorted_graph):
            if t._ctx is None:
                continue

            backward_fn, arg_ctx = t._ctx
            inputs = arg_ctx.get("inputs", [])  # Use .get for safety

            upstream_grad = t.grad
            if upstream_grad is None:
                continue

            input_grads = backward_fn(upstream_grad, arg_ctx)

            # Ensure input_grads is a tuple, as autograd functions should return tuples
            if not isinstance(input_grads, tuple):
                raise TypeError(
                    f"Backward function {backward_fn.__name__} must return a tuple of gradients."
                )

            # This zip correctly pairs each input tensor with its calculated gradient.
            for parent_tensor, grad_for_parent in zip(inputs, input_grads):
                if isinstance(parent_tensor, Tensor) and parent_tensor.requires_grad:
                    if parent_tensor.grad is None:
                        parent_tensor.grad = np.zeros_like(parent_tensor.data)
                    parent_tensor.grad += grad_for_parent

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        if self._ctx is not None:
            _, ctx = self._ctx
            for prev in ctx["inputs"]:
                prev.zero_grad()

    def __repr__(self) -> str:
        return f"{self.data}, dtype={self.dtype}"
