# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from PIL import Image, ImageFilter
from typing import Any, Dict, List, Optional, Tuple, Union
from unitorch.utils import is_opencv_available
from unitorch.cli import CoreConfigureParser

from unitorch.cli.pipelines.stable_flux.text2image import (
    StableFluxForText2ImageGenerationPipeline,
)
from unitorch.cli.pipelines.stable_flux.image2image import (
    StableFluxForImage2ImageGenerationPipeline,
)
from unitorch.cli.pipelines.stable_flux.image_control import (
    StableFluxForImageControlGenerationPipeline,
)
from unitorch.cli.pipelines.stable_flux.image_redux import (
    StableFluxForImageReduxGenerationPipeline,
)
from unitorch.cli.pipelines.stable_flux.inpainting import (
    StableFluxForImageInpaintingPipeline,
)
