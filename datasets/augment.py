import os
import sys
import math
import random

import cv2
import numpy as np

sys.path.insert(1, os.path.abspath('..'))
from auxiliary.utils import rgb_to_bgr

class DataAugmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        pass


    def crop(self, img: np.ndarray, scale: float = 0.5) -> np.ndarray:
        if self.cfg.mode == "train" or "valid":
            return cv2.resize(img, (self.cfg.input_size.train.w, self.cfg.input_size.train.h), fx=scale, fy=scale)
        else:
            return cv2.resize(img, (self.cfg.input_size.test.w, self.cfg.fcn_input_size.test.h), fx=scale, fy=scale)


    def augment(self, img: np.ndarray, illum: np.ndarray) -> tuple:
        img = self.crop(img, 0.5)
        return img, illum




