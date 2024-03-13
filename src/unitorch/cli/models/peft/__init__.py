# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

pretrained_peft_infos = {}

import unitorch.cli.models.peft.modeling_bloom
import unitorch.cli.models.peft.modeling_llama
import unitorch.cli.models.peft.modeling_mistral
from unitorch.cli.models.peft.modeling_bloom import (
    BloomLoraForClassification,
    BloomLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_llama import (
    LlamaLoraForClassification,
    LlamaLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_mistral import MistralLoraForGeneration
