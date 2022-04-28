"""import torch

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

import cv2
import numpy as np
from Data.for_infer import LoadStreams

source = "0"
dataset = LoadStreams(source)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.299, 0.224, 0.225])
for source, img, img0, cap, s, rp in dataset:
    print(img.shape)
    if len(img0) == 1:
        img0 = img0[0]
    print(img0.shape)
    img = img[0].transpose(1, 2, 0)[..., ::-1]
    #img *= std
    #img += mean
    print(img)
    cv2.imshow("img", img)
    cv2.waitKey(1)
"""

import numpy as np

a = np.array([
    [1, 3, 4, 5, 6],
    [3, 2, 1, 4, 6],
    [2, 45, 5, 2, 1]
])
print(a[:, 2])