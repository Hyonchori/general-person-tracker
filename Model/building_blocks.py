# Convolution blocks used in deep learning architecture
# Refer to YOLOx: https://github.com/Megvii-BaseDetection/YOLOX
from typing import List

import torch
import torch.nn as nn


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError(f"Unssurported activation function type: {name}")
    return module


class BaseConv(nn.Module):
    """
    Normal: Conv2d -> BatchNorm -> SiLU/LeakyRelU
    Fused: FusedConv2d -> SiLU/LeakeyRelU
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ksize: int,
                 stride: int,
                 groups: int = 1,
                 bias: bool = False,
                 act: str = "silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """ Depth-wise Conv + Conv """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ksize: int,
                 stride: int = 1,
                 act: str = "silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act
        )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class BottleNeck(nn.Module):
    # Standard Bottleneck
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = True,
                 expansion: float = 0.5,
                 depthwise: bool = False,
                 act: str = "silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleNeck(nn.Module):
    """ Spatial Pyramid Pooling layer used in YOLOv3-SPP """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ksizes: List[int] = (5, 9, 13),
                 act: str = "silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in ksizes
        ])
        conv2_channels = hidden_channels * (len(ksizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """
    C3 in YOLOv5, CSP BottleNeck with 3 convolutions
    n(int): number of Bottlenecks. Default value = 1
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 shortcut: bool = True,
                 expansion: float = 0.5,
                 depthwise: bool = False,
                 act: str = "silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            BottleNeck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            ) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.m(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """ Focus width and height information into channel space """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ksize: int = 1,
                 stride: int = 1,
                 act: str = "silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x(b, c, w, h) -> y(b, 4c, w/2, h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bottom_left = x[..., 1::2, ::2]
        patch_bottom_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bottom_left,
                patch_top_right,
                patch_bottom_right
            ),
            dim=1
        )
        return self.conv(x)
