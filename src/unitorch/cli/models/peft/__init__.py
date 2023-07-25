# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch.cli.models.peft.modeling_bloom
import unitorch.cli.models.peft.modeling_llama
from unitorch.cli.models.peft.modeling_bloom import (
    BloomAdaLoraForClassification,
    BloomAdaLoraForGeneration,
    BloomLoraForClassification,
    BloomLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_llama import (
    LlamaAdaLoraForClassification,
    LlamaAdaLoraForGeneration,
    LlamaLoraForClassification,
    LlamaLoraForGeneration,
)
from unitorch.cli.models.peft.modeling_minigpt4 import MiniGPT4LoraForGeneration
