from abc import ABC, abstractmethod
import neura

class Module(ABC):
    @abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError("Forward method is not implemented yet.")
    
    @abstractmethod
    def backward(self, x, **kwargs):
        raise NotImplementedError("Backward method is not implemented yet.")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        params = []
        for _, value in vars(self).items():
            if isinstance(value, neura.Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params
