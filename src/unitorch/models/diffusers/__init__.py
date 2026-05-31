# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available
from unitorch.models.diffusers.modeling_wan import (
    WanForText2VideoGeneration,
    WanForImage2VideoGeneration,
)
from unitorch.models.diffusers.modeling_qwen_image import (
    GenericQWenImageModel,
    QWenImageText2ImageGeneration,
    QWenImageEditingGeneration,
)
from unitorch.models.diffusers.modeling_flux2 import (
    GenericFlux2Model,
    Flux2ForText2ImageGeneration,
    Flux2ForImageEditingGeneration,
)
from unitorch.models.diffusers.modeling_lucy import LucyForVideoEditingGeneration
from unitorch.models.diffusers.modeling_vae import VAEForDiffusion
from unitorch.models.diffusers.processing_qwen_image import QWenImageProcessor
from unitorch.models.diffusers.processing_flux2 import Flux2Processor

if is_opencv_available():
    from unitorch.models.diffusers.processing_wan import WanProcessor
    from unitorch.models.diffusers.processing_lucy import LucyProcessor
