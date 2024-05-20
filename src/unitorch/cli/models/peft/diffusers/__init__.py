# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch.cli.models.peft.diffusers.modeling_controlnet_xl
import unitorch.cli.models.peft.diffusers.modeling_controlnet
import unitorch.cli.models.peft.diffusers.modeling_stable_xl
import unitorch.cli.models.peft.diffusers.modeling_stable
from unitorch.cli.models.peft.diffusers.modeling_controlnet_xl import (
    ControlNetXLLoraForText2ImageGeneration,
    ControlNetXLLoraForImage2ImageGeneration,
    ControlNetXLLoraForImageInpainting,
)
from unitorch.cli.models.peft.diffusers.modeling_controlnet import (
    ControlNetLoraForText2ImageGeneration,
    ControlNetLoraForImage2ImageGeneration,
    ControlNetLoraForImageInpainting,
)
from unitorch.cli.models.peft.diffusers.modeling_stable_xl import (
    StableXLLoraForText2ImageGeneration,
    StableXLLoraForImage2ImageGeneration,
    StableXLLoraForImageInpainting,
)
from unitorch.cli.models.peft.diffusers.modeling_stable import (
    StableLoraForText2ImageGeneration,
    StableLoraForImage2ImageGeneration,
    StableLoraForImageInpainting,
    StableLoraForImageResolution,
)
