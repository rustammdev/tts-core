# Minimal reimplementation of torchvision.ops.misc.ConvNormActivation
# (BSD-3-Clause, https://github.com/pytorch/vision) — vendored to avoid a
# full torchvision dependency; state_dict layout is identical.
import torch.nn as nn


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        dilation=1,
        inplace=True,
        bias=None,
    ):
        if padding is None:
            padding = tuple(
                (k - 1) // 2 * d for k, d in zip(_pair(kernel_size), _pair(dilation))
            )
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels
