# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

pretrained_peft_infos = {}

from unitorch.utils import is_diffusers_available

import unitorch.cli.models.peft.modeling_bloom
import unitorch.cli.models.peft.modeling_clip
import unitorch.cli.models.peft.modeling_llama
import unitorch.cli.models.peft.modeling_mistral

if is_diffusers_available():
    import unitorch.cli.models.peft.diffusers
from unitorch.cli.models.peft.modeling_bloom import (
    BloomLoraForClassification,
    BloomLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_llama import (
    LlamaLoraForClassification,
    LlamaLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_llava import (
    LlavaMistralClipLoraForClassification,
    LlavaMistralClipLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_mistral import (
    MistralLoraForClassification,
    MistralLoraForGeneration,
)
