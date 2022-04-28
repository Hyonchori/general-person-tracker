# torch Dataset for train detection model
import os
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO


class COCODataset(Dataset):
    """
    COCO dataset class.
    Images should be located in 'data_dir'
    """
    def __init__(self,
                 data_dir: str = None,
                 json_path: str = None,
                 img_size: Tuple[int] = (720, 1280),
                 preproc=None):
        super().__init__()
        self.data_dir = data_dir
        self.json_path = json_path

        self.coco = COCO(self.json_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._class = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = obj["bbox"][2]
            y2 = obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for i, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[i, 0: 4] = obj["clean_bbox"]
            res[i, 4] = cls

        file_name = im_ann["file_name"] if "file_name" in im_ann else f"{id_:012}.jpg"
        img_info = (height, width, file_name)

        del im_ann, annotations
        return res, img_info, file_name

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]
        img_path = os.path.join(self.data_dir, file_name)
        img = cv2.imread(img_path)
        assert img is not None
        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.img_size)
        return img, target, img_info, img_id


if __name__ == "__main__":
    val_dir = "/home/daton/Downloads/coco/val2017"
    val_path = "/home/daton/Downloads/coco/annotations_trainval2017/annotations/instances_val2017.json"
    dataset = COCODataset(val_dir, val_path)

    for img, target, img_info, img_id in dataset:
        print(img.shape)
        break
