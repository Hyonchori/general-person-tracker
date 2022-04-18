# Backbone of YOLOx: CSPDarknet + PAFPN
from typing import List

import torch
import torch.nn as nn

from .base_neck import BaseNeck
from ..backbones.base_backbone import BaseBackbone
from ..backbones.darknet import CSPDarknet
from ..building_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(BaseNeck):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    def __init__(self,
                 backbone: BaseBackbone = None,
                 depth: float = 1.0,
                 width: float = 1.0,
                 in_features: List[str] = ("dark3", "dark4", "dark5"),
                 in_channels: List[int] = (256, 512, 1024),
                 out_features: List[str] = ("pan2", "pan1", "pan0"),
                 depthwise: bool = False,
                 act: str = "silu"):
        if backbone is None:
            backbone = CSPDarknet(depth, width, out_features=in_features, depthwise=depthwise, act=act)
        super().__init__(backbone=backbone,
                         in_features=in_features,
                         out_features=out_features)
        assert len(in_features) == len(in_channels) == 3, "Length of in_features and in_channels should be 3"
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

    def forward(self, x):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        outputs = {}
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        outputs["pan2"] = pan_out2

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        outputs["pan1"] = pan_out1

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        outputs["pan0"] = pan_out0

        outputs = [v for k, v in outputs.items() if k in self.out_features]
        return outputs
