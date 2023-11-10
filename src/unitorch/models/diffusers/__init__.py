# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch import is_diffusers_available

if is_diffusers_available():
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
    )
    from unitorch.models.diffusers.modeling_stable_xl_refiner import (
        StableXLRefinerForText2ImageGeneration,
        StableXLRefinerForImage2ImageGeneration,
        StableXLRefinerForImageInpainting,
    )
    from unitorch.models.diffusers.modeling_dreambooth import (
        DreamboothForText2ImageGeneration,
    )
    from unitorch.models.diffusers.modeling_dreambooth_xl import (
        DreamboothXLForText2ImageGeneration,
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
    from unitorch.models.diffusers.modeling_multicontrolnet import (
        MultiControlNetForText2ImageGeneration,
        MultiControlNetForImage2ImageGeneration,
        MultiControlNetForImageInpainting,
    )
    from unitorch.models.diffusers.modeling_animate import (
        AnimateForText2VideoGeneration,
        AnimateForImage2VideoGeneration,
    )
    from unitorch.models.diffusers.modeling_blip2 import Blip2ForText2ImageGeneration
    from unitorch.models.diffusers.processing_stable import StableProcessor
    from unitorch.models.diffusers.processing_stable_xl import StableXLProcessor
    from unitorch.models.diffusers.processing_stable_xl_refiner import (
        StableXLRefinerProcessor,
    )
    from unitorch.models.diffusers.processing_dreambooth import DreamboothProcessor
    from unitorch.models.diffusers.processing_dreambooth_xl import DreamboothXLProcessor
    from unitorch.models.diffusers.processing_controlnet import ControlNetProcessor
    from unitorch.models.diffusers.processing_controlnet_xl import ControlNetXLProcessor
    from unitorch.models.diffusers.processing_multicontrolnet import (
        MultiControlNetProcessor,
    )
    from unitorch.models.diffusers.processing_blip2 import Blip2Processor
