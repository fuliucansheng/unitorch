# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

pretrained_peft_infos = {}

from unitorch.utils import is_diffusers_available

import unitorch.cli.models.peft.modeling_bloom
import unitorch.cli.models.peft.modeling_clip
import unitorch.cli.models.peft.modeling_llama
import unitorch.cli.models.peft.modeling_llava
import unitorch.cli.models.peft.modeling_mistral
import unitorch.cli.models.peft.modeling_qwen
import unitorch.cli.models.peft.modeling_qwen_vl

if is_diffusers_available():
    import unitorch.cli.models.peft.diffusers
