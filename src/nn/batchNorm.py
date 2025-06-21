import neura
import numpy as np

from .module import Module


class BatchNorm2d(Module):
    def __init__(self, m: int, epsilon: float = 1e-5):
        self.m = m
        self.epsilon = epsilon
        self.gamma = neura.Tensor(np.zeros((1, m)))
        self.beta = neura.Tensor(np.zeros((1, m)))

    def forward(self, x: neura.Tensor) -> neura.Tensor:
        mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
        variance = np.mean((x.data - mean) ** 2, axis=(0, 2, 3), keepdims=True)

        x_bar = (x.data - mean) / np.sqrt(variance + self.epsilon)
        x_bar = neura.Tensor(x_bar)

        out = self.gamma @ x_bar + self.beta

        return out
