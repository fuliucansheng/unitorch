# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, List, Optional, Tuple, Union
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.dpt import DPTForDepthEstimationPipeline

dpt_pipe = None


def depth(image: Image.Image):
    global dpt_pipe
    if dpt_pipe is None:
        dpt_pipe = DPTForDepthEstimationPipeline.from_core_configure(
            CoreConfigureParser(), pretrained_name="dpt-large"
        )
        dpt_pipe.to("cpu")
    return dpt_pipe(image)
