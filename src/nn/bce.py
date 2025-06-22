import neura

from .module import Module


class BCEWithLogitLoss(Module):
    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")

    def forward(self, x: neura.Tensor, target: neura.Tensor) -> neura.Tensor:
        max_val = (-x).relu()
        loss = x - x * target + max_val + ((-x).abs() + max_val).exp().log()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
