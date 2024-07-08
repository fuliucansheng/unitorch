# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, List, Optional, Tuple, Union
from unitorch.utils import is_opencv_available
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.dpt import DPTForDepthEstimationPipeline


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


dpt_pipe = None


def depth(image: Image.Image):
    global dpt_pipe
    if dpt_pipe is None:
        dpt_pipe = DPTForDepthEstimationPipeline.from_core_configure(
            CoreConfigureParser(), pretrained_name="dpt-large"
        )
        dpt_pipe.to("cpu")
    return dpt_pipe(image)


controlnet_processes = {
    "canny": canny,
    # "depth": depth,
}

from unitorch.cli.pipelines.stable_3.text2image import (
    Stable3ForText2ImageGenerationPipeline,
)
from unitorch.cli.pipelines.stable_3.image2image import (
    Stable3ForImage2ImageGenerationPipeline,
)
