# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, List, Optional, Tuple, Union
from unitorch.utils import is_opencv_available


def canny(image: Image.Image):
    if is_opencv_available():
        import cv2

        image = np.array(image, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.Canny(image, 100, 200)
        image = Image.fromarray(image)
    else:
        image = image.convert("L")
        image = image.filter(ImageFilter.FIND_EDGES)
    return image
