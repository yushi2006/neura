import neura
import numpy as np

from .module import Module


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = neura.Tensor(np.random.randn(in_dim, out_dim))
        self.b = neura.Tensor(np.zeros(out_dim))

    def forward(self, x: neura.Tensor) -> neura.Tensor:
        self.x = x
        return x @ self.W + self.b
