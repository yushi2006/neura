import numpy as np

from ..core import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = Tensor(np.random.randn(in_dim, out_dim))
        self.b = Tensor(np.zeros(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return x @ self.W + self.b
