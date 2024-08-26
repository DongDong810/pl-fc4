import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(1, os.path.abspath('..'))
from auxiliary.utils import get_mcc_coord, load_image, load_image_without_mcc, normalize, bgr_to_rgb, correct


"""
All images in the Color Checker dataset are linear images in the RAW format of the acquisition device, each with a
Macbeth ColorChecker (MCC) chart, which provides an estimation of illuminant colors. To prevent the CNN from detecting
and utilizing MCCs as a visual cue, all images are masked with provided locations of MCC during training and testing
"""

BASE_PATH = "preprocessed"

# Existing files
PATH_TO_IMAGES = os.path.join("images")
PATH_TO_COORDINATES = os.path.join("coordinates")
PATH_TO_CC_METADATA = os.path.join("metadata.txt")

# New files
PATH_TO_NUMPY_DATA = os.path.join(BASE_PATH, "numpy_data")
PATH_TO_NUMPY_LABELS = os.path.join(BASE_PATH, "numpy_labels")
PATH_TO_LINEAR_IMAGES = os.path.join(BASE_PATH, "linear_images")
PATH_TO_GT_CORRECTED = os.path.join(BASE_PATH, "gt_corrected")

def main():
    print("\n=================================================\n")
    print("\t Masking MCC charts")
    print("\n=================================================\n")






    # Generating for pre-processing dataset





    return

if __name__ == "__main__":
    main()
