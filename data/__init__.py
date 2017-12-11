from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .mpii import MpiiDetection, MpiiAnnotationTransform, MPII_CLASSES

from .config import *
import cv2
import numpy as np


def base_transform(image, width, height, mean):
    x = cv2.resize(image, (int(width), int(height))).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, height, width, mean):
        self.height = height
        self.width = width
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, width=self.width, height=self.height, mean = self.mean), boxes, labels
