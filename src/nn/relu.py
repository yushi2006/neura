from ..core import Tensor
from .module import Module


class ReLU(Module):
    def __init__(self, x):
        self.x = x

    def forward(self) -> Tensor:
        return self.x.relu()
