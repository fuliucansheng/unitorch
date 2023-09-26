# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch import is_diffusers_available

if is_diffusers_available():
    from unitorch.models.diffusers.modeling_controlnet import (
        ControlNetForText2ImageGeneration,
        ControlNetForImage2ImageGeneration,
        ControlNetForImageInpainting,
    )
    from unitorch.models.diffusers.processing_controlnet import ControlNetProcessor
    from unitorch.models.diffusers.processing_controlnet_xl import ControlNetXLProcessor
    from unitorch.models.diffusers.processing_stable import StableProcessor
    from unitorch.models.diffusers.processing_stable_xl import StableXLProcessor
    from unitorch.models.diffusers.modeling_stable import (
        StableForText2ImageGeneration,
        StableForImage2ImageGeneration,
        StableForImageInpainting,
        StableForImageResolution,
    )
    from unitorch.models.diffusers.modeling_stable_xl import (
        StableXLForText2ImageGeneration,
        StableXLForImage2ImageGeneration,
        StableXLForImageInpainting,
        StableXLRefinerForText2ImageGeneration,
    )
    from unitorch.models.diffusers.modeling_controlnet_xl import (
        ControlNetXLForText2ImageGeneration,
        ControlNetXLForImage2ImageGeneration,
        ControlNetXLForImageInpainting,
    )
