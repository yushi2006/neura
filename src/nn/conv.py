from typing import Union

import neura
import neura.nn as nn
import numpy as np


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[tuple, int] = 0,
        stride: Union[tuple, int] = 1,
        has_bias: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.has_bias = has_bias

        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        scale = np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.W = neura.Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * scale,
            requires_grad=True,
        )

        self.b = (
            neura.Tensor(np.zeros((self.out_channels, 1, 1))) if self.has_bias else None
        )

    def forward(self, x: neura.Tensor) -> neura.Tensor:
        _, C_in, _, _ = x.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Input tensor channels {C_in} ≠ expected {self.in_channels}"
            )

        out = x.conv2d(self.W, padding=self.padding, stride=self.stride)

        if self.b is not None:
            out = out + self.b
        return out


class Conv2dTranpose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[tuple, int] = 0,
        stride: Union[tuple, int] = 1,
        has_bias: bool = False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.has_bias = has_bias

        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        scale = np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.W = neura.Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * scale,
            requires_grad=True,
        )

        self.b = (
            neura.Tensor(np.zeros((self.out_channels, 1, 1))) if self.has_bias else None
        )

    def forward(self, x: neura.Tensor) -> neura.Tensor:
        _, C_in, _, _ = x.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Input tensor channels {C_in} ≠ expected {self.in_channels}"
            )

        out = x.conv2dTranspose(self.W, padding=self.padding, stride=self.stride)

        if self.has_bias:
            out += self.b

        return out
