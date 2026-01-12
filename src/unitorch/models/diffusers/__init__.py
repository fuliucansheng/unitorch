# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available
from unitorch.models.diffusers.modeling_stable import compute_snr
from unitorch.models.diffusers.modeling_stable import (
    GenericStableModel,
    StableForText2ImageGeneration,
    StableForImage2ImageGeneration,
    StableForImageInpainting,
    StableForImageResolution,
    StableForImage2VideoGeneration,
)
from unitorch.models.diffusers.modeling_stable_xl import (
    GenericStableXLModel,
    StableXLForText2ImageGeneration,
    StableXLForImage2ImageGeneration,
    StableXLForImageInpainting,
)

from unitorch.models.diffusers.modeling_stable_3 import (
    GenericStable3Model,
    Stable3ForText2ImageGeneration,
    Stable3ForImage2ImageGeneration,
    Stable3ForImageInpainting,
)
from unitorch.models.diffusers.modeling_stable_flux import (
    GenericStableFluxModel,
    StableFluxForText2ImageGeneration,
    StableFluxForImage2ImageGeneration,
    StableFluxForImageReduxGeneration,
    StableFluxForImageInpainting,
    StableFluxForKontext2ImageGeneration,
)
from unitorch.models.diffusers.modeling_wan import (
    WanForText2VideoGeneration,
    WanForImage2VideoGeneration,
)
from unitorch.models.diffusers.modeling_qwen_image import (
    GenericQWenImageModel,
    QWenImageText2ImageGeneration,
    QWenImageEditingGeneration,
)
from unitorch.models.diffusers.modeling_vae import VAEForDiffusion
from unitorch.models.diffusers.processing_stable import (
    StableProcessor,
    StableVideoProcessor,
)
from unitorch.models.diffusers.processing_stable_xl import StableXLProcessor
from unitorch.models.diffusers.processing_stable_3 import Stable3Processor
from unitorch.models.diffusers.processing_stable_flux import StableFluxProcessor
from unitorch.models.diffusers.processing_qwen_image import QWenImageProcessor

if is_opencv_available():
    from unitorch.models.diffusers.processing_wan import WanProcessor
