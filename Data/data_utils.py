import re

import cv2
import numpy as np

from typing import Tuple


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
        bboxes=None
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = new_shape + (new_shape + stride) % stride
        new_shape = (new_shape, new_shape)
    else:
        new_shape = [x + (x + stride) % stride for x in new_shape]

    if im.shape[:2] == new_shape:
        return im, 1., (0, 0)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if bboxes is None:
        return im, ratio, (dw, dh)
    else:
        bboxes *= ratio
        bboxes[:, 0] += dw
        bboxes[:, 2] += dw
        bboxes[:, 1] += dh
        bboxes[:, 3] += dh
        return im, ratio, (dw, dh), bboxes


def preproc(img: np.ndarray,
            mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
            std: np.ndarray = np.array([0.299, 0.224, 0.225]),
            swap: Tuple[int] = (2, 0, 1),
            scaling: bool = True,
            normalize: bool = True,
            brg2rgb: bool = True):
    img = img.astype(np.float32)
    if scaling or normalize:
        img /= 255.0
    if normalize:
        img -= mean
        img /= std
    if brg2rgb:
        img = img[..., ::-1]
    if swap:
        img = img.transpose(swap)
    return img


if __name__ == "__main__":
    img = np.zeros((480, 640, 3))
    img = letterbox(img, 640)[0]
    print(img.shape)
    cv2.imshow("img", img)
    cv2.waitKey(0)
