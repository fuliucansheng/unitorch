# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, List, Optional, Tuple, Union
from unitorch.utils import is_opencv_available

from unitorch.cli.pipelines.wan.text2video import (
    WanForText2VideoGenerationPipeline,
)
from unitorch.cli.pipelines.wan.image2video import (
    WanForImage2VideoGenerationPipeline,
)
