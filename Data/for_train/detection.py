# torch Dataset for train detection model
import os
from typing import List

import cv2
from torch.utils.data.dataset import Dataset

from pycocotools.coco import COCO


class COCODataset(Dataset):
    """
    COCO dataset class.
    Images should be located in 'data_dir'
    """
    def __init__(self,
                 data_dir: str=None,
                 json_path: str=None,
                 img_size: List[int]=(720, 1280),
                 preproc=None):
        super().__init__()
        