from abc import ABC, abstractmethod

from neura import Tensor


class Module(ABC):
    @abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError("Forward method is not implemented yet.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for _, value in vars(self).items():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params
