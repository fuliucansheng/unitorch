# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available
from unitorch.models.diffusers.modeling_stable_flux import (
    compute_snr,
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
from unitorch.models.diffusers.processing_stable_flux import StableFluxProcessor
from unitorch.models.diffusers.processing_qwen_image import QWenImageProcessor

if is_opencv_available():
    from unitorch.models.diffusers.processing_wan import WanProcessor
