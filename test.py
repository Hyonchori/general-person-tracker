'''import torch

from Model.backbones.darknet import CSPDarknet
from Model.necks.yolo_pafpn import YOLOPAFPN
from Model.heads.yolox_head import YOLOXHead
from Model.yolox import YOLOX
from Model.model_utils import model_info

bb = CSPDarknet()
nn = YOLOPAFPN()
hh = YOLOXHead(80)

input_sample = [torch.randn((8, 256, 736 // 8, 1280 // 8)),
                torch.randn((8, 512, 736 // 16, 1280 // 16)),
                torch.randn((8, 1024, 736 // 32, 1280 // 32))]

input_sample = torch.randn((2, 3, 736, 1280))

model = YOLOX()
outputs = model(input_sample)

model_info(model, verbose=True)
'''
import cv2
from Data.for_infer import LoadStreams

source = "0"
dataset = LoadStreams(source)
for source, img, img0, cap, s, rp in dataset:
    print(img.shape)
    if len(img0) == 1:
        img0 = img0[0]
    print(img0.shape)
    cv2.imshow("img", img0)
    cv2.waitKey(1)
