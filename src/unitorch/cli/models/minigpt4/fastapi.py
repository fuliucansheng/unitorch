# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.minigpt4 import (
    MiniGPT4Blip2LlamaForGeneration as _MiniGPT4Blip2LlamaForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_fastapi,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.minigpt4 import pretrained_minigpt4_infos



@register_fastapi("core/fastapi/generation/minigpt4")
class MiniGPT4Blip2LlamaForGeneration(_MiniGPT4Blip2LlamaForGeneration):
    def fastapi(self,):
        pass