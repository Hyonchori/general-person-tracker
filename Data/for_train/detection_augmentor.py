# Augmented dataset for train detector
import random

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

from .data_augment import box_candidates, random_perspective


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class DetectionAugmentor(Dataset):
    """
    Detection dataset that performs mosaic and mixup for normal dataset.
    """

    def __init__(self, dataset, img_size, mosaic=True, preproc=None,
                 degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
                 shear=2.0, perspective=0.0, mixup=0.5, enable_mixup=True, *args):
        super().__init__()
        self.img_size = img_size
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup = mixup
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels = []
            input_size = self._dataset.img_size
            input_h, input_w = input_size[0], input_size[1]

            # mosaic center
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i, index in enumerate(indices):
                img, _labels, _, _ = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate mosaic image
                (h, w, c) = img.shape[:3]
                if i == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l: large image, s: small image in mosaic image
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i, xc, yc, w, h, input_h, input_w
                )
                mosaic_img[l_y1: l_y2, l_x1: l_x2] = img[s_y1: s_y2, s_x1: s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_img, 0)
                mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < 2 * input_w]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < 2 * input_h]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )

            if self.enable_mosaic and not len(mosaic_labels) == 0 and :
                mosaic_img, mosaic_labels = self.mixup_scale(mosaic_img, mosaic_labels, self.img_size)

            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.img_size)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            return mix_img, padded_labels, img_info, np.array([idx])

        else:
            self._dataset._img_size = self.img_size
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.img_size)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_size):
        jit_factor = random.uniform(*self.mixup_scale)
        flip = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_size[0], input_size[1], 3)) * 114.
        else:
            cp_img = np.ones(input_size) * 114.
        cp_scale_ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor))
        )
        cp_scale_ratio *= jit_factor
        if flip:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[: 2]
        target_h, target_w = origin_img.shape[: 2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

