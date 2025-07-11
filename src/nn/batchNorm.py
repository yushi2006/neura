import numpy as np

from ..core import Tensor
from .module import Module


class BatchNorm2d(Module):
    def __init__(self, m: int, epsilon: float = 1e-5):
        super().__init__()  # if Module requires init
        self.m = m
        self.epsilon = epsilon
        # Initialize gamma=1 and beta=0 with shape (1, C, 1, 1)
        gamma_np = np.ones((1, m, 1, 1), dtype=float)
        beta_np = np.zeros((1, m, 1, 1), dtype=float)
        self.gamma = Tensor(gamma_np)
        self.beta = Tensor(beta_np)

    def forward(self, x: Tensor) -> Tensor:
        x_np = x.data
        if x_np.ndim != 4:
            raise ValueError(f"BatchNorm2d.forward expects 4D input, got {x_np.shape}")
        N, C, H, W = x_np.shape
        if C != self.m:
            raise ValueError(
                f"Channel mismatch: input has C={C}, but BatchNorm2d was created with m={self.m}"
            )

        # compute mean & variance over (N, H, W) for each channel
        mean = np.mean(x_np, axis=(0, 2, 3), keepdims=True)  # shape (1, C, 1, 1)
        variance = np.mean(
            (x_np - mean) ** 2, axis=(0, 2, 3), keepdims=True
        )  # shape (1, C, 1, 1)

        x_bar_np = (x_np - mean) / np.sqrt(
            variance + self.epsilon
        )  # shape (N, C, H, W)
        x_bar = Tensor(x_bar_np)

        # Prepare gamma for broadcasting.
        gamma = self.gamma
        # Common incorrect shapes to handle: (C,), (1, C), (C,1,1), etc.
        if gamma.shape != (1, C, 1, 1):
            gs = gamma.shape
            # If shape == (C,)
            if len(gs) == 1 and gs[0] == C:
                reshaped = gamma.data.reshape((1, C, 1, 1))
                gamma = Tensor(reshaped)
            # If shape == (1, C)
            elif len(gs) == 2 and gs[0] == 1 and gs[1] == C:
                reshaped = gamma.data.reshape((1, C, 1, 1))
                gamma = Tensor(reshaped)
            # If shape == (C, 1, 1)
            elif len(gs) == 3 and gs[0] == C and gs[1] == 1 and gs[2] == 1:
                reshaped = gamma.data.reshape((1, C, 1, 1))
                gamma = Tensor(reshaped)
            # If shape already 4D but not matching (1,C,1,1), e.g., (N,C,H,W): cannot reshape learnable param
            elif len(gs) == 4 and gs[1] == C and gs.count(1) == 3:
                # shape might be (1,C,1,1) in different ordering—but unlikely; skip
                # If shape matches broadcastable to (N,C,H,W)? e.g., (1,C,1,1) is correct.
                gamma = gamma  # keep as is
            else:
                # Generic attempt: try reshape to (1,C,1,1) if total elements match C
                try:
                    if np.prod(gs) == C:
                        reshaped = gamma.data.reshape((1, C, 1, 1))
                        gamma = Tensor(reshaped)
                    else:
                        # Cannot reshape in a meaningful way
                        raise ValueError
                except Exception:
                    raise ValueError(
                        f"Cannot reshape gamma of shape {gs} to (1, {C}, 1, 1)"
                    )
        # Now gamma has shape (1, C, 1, 1). Broadcast to (N, C, H, W).
        if gamma.shape != x_bar.shape:
            try:
                gamma = gamma.broadcast_to((N, C, H, W))
            except Exception as e:
                raise ValueError(
                    f"Cannot broadcast gamma of shape {gamma.shape} to {(N, C, H, W)}: {e}"
                )

        # Prepare beta similarly
        beta = self.beta
        if beta.shape != (1, C, 1, 1):
            bs = beta.shape
            if len(bs) == 1 and bs[0] == C:
                reshaped = beta.data.reshape((1, C, 1, 1))
                beta = Tensor(reshaped)
            elif len(bs) == 2 and bs[0] == 1 and bs[1] == C:
                reshaped = beta.data.reshape((1, C, 1, 1))
                beta = Tensor(reshaped)
            elif len(bs) == 3 and bs[0] == C and bs[1] == 1 and bs[2] == 1:
                reshaped = beta.data.reshape((1, C, 1, 1))
                beta = Tensor(reshaped)
            elif len(bs) == 4 and bs[1] == C and bs.count(1) == 3:
                beta = beta
            else:
                try:
                    if np.prod(bs) == C:
                        reshaped = beta.data.reshape((1, C, 1, 1))
                        beta = Tensor(reshaped)
                    else:
                        raise ValueError
                except Exception:
                    raise ValueError(
                        f"Cannot reshape beta of shape {bs} to (1, {C}, 1, 1)"
                    )
        if beta.shape != x_bar.shape:
            try:
                beta = beta.broadcast_to((N, C, H, W))
            except Exception as e:
                raise ValueError(
                    f"Cannot broadcast beta of shape {beta.shape} to {(N, C, H, W)}: {e}"
                )

        # Finally, elementwise scale and shift
        out = x_bar * gamma + beta
        return out
