# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, List, Optional, Tuple, Union
from unitorch.utils import is_opencv_available

from unitorch.cli.pipelines.stable.text2image import (
    StableForText2ImageGenerationPipeline,
)
from unitorch.cli.pipelines.stable.image2image import (
    StableForImage2ImageGenerationPipeline,
)
from unitorch.cli.pipelines.stable.inpainting import StableForImageInpaintingPipeline
from unitorch.cli.pipelines.stable.resolution import StableForImageResolutionPipeline
from unitorch.cli.pipelines.stable.image2video import (
    StableForImage2VideoGenerationPipeline,
)
from unitorch.cli.pipelines.stable.interrogator import ClipInterrogatorPipeline
