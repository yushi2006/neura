import neura

from .module import Module


class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x: neura.Tensor) -> neura.Tensor:
        return x.relu()
