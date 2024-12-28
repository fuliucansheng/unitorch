# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

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
    StableFluxForImageControlGeneration,
    StableFluxForImageReduxGeneration,
    StableFluxForImageInpainting,
)
from unitorch.models.diffusers.modeling_controlnet import (
    ControlNetForText2ImageGeneration,
    ControlNetForImage2ImageGeneration,
    ControlNetForImageInpainting,
)
from unitorch.models.diffusers.modeling_controlnet_xl import (
    ControlNetXLForText2ImageGeneration,
    ControlNetXLForImage2ImageGeneration,
    ControlNetXLForImageInpainting,
)

from unitorch.models.diffusers.modeling_controlnet_3 import (
    ControlNet3ForText2ImageGeneration,
    ControlNet3ForImageInpainting,
)
from unitorch.models.diffusers.modeling_controlnet_flux import (
    ControlNetFluxForText2ImageGeneration,
    ControlNetFluxForImage2ImageGeneration,
    ControlNetFluxForImageInpainting,
)
from unitorch.models.diffusers.modeling_adapter import (
    StableAdapterForText2ImageGeneration,
)
from unitorch.models.diffusers.modeling_adapter_xl import (
    StableXLAdapterForText2ImageGeneration,
)
from unitorch.models.diffusers.modeling_vae import VAEForDiffusion
from unitorch.models.diffusers.processing_stable import (
    StableProcessor,
    StableVideoProcessor,
)
from unitorch.models.diffusers.processing_stable_xl import StableXLProcessor
from unitorch.models.diffusers.processing_stable_3 import Stable3Processor
from unitorch.models.diffusers.processing_stable_flux import StableFluxProcessor
