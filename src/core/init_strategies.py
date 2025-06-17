import numpy as np
from abc import ABC, abstractmethod


class InitStrategy(ABC):
    @abstractmethod
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        pass


class ZeroInit(ABC):
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class OneInit(ABC):
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)


class RandnInit(ABC):
    def init(self, shape: tuple, dtype: type = np.float32) -> np.ndarray:
        return np.random.random(*shape).astype(dtype)
