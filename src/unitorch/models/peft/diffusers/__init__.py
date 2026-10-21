# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils.import_utils import is_flux3_available

if is_flux3_available():
    from unitorch.models.peft.diffusers.modeling_flux3 import (
        Flux3LoraForText2ImageGeneration, Flux3LoraForImageEditingGeneration,
    )

from unitorch.models.peft.diffusers.modeling_qwen_image import (
    QWenImageLoraForText2ImageGeneration,
    QWenImageLoraForImageEditing,
)
from unitorch.models.peft.diffusers.modeling_wan import (
    WanLoraForText2VideoGeneration,
    WanLoraForImage2VideoGeneration,
)
from unitorch.models.peft.diffusers.modeling_flux2 import (
    Flux2LoraForText2ImageGeneration,
    Flux2LoraForImageEditingGeneration,
)
from unitorch.models.peft.diffusers.modeling_lucy import (
    LucyLoraForVideoEditingGeneration,
)
