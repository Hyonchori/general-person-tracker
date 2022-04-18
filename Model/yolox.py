# YOLOx for object detection
import torch.nn as nn

from .necks.yolo_pafpn import YOLOPAFPN
from .heads.yolox_head import YOLOXHead


class YOLOX(nn.Module):
    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)  # for COCO dataset
        assert len(backbone.out_features) == len(head.in_channels) == len(head.strides)
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        for o in fpn_outs:
            print(o.shape)
        return self.head(fpn_outs)
