# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast

from unitorch.models.diffusers import VAEForDiffusion as _VAEForDiffusion
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)


@register_model("core/model/diffusers/vae")
class VAEForDiffusion(_VAEForDiffusion):
    def __init__(
        self,
        config_path: str,
        patch_size: Optional[int] = 32,
        stride: Optional[int] = 16,
        use_fp16: Optional[bool] = True,
        use_bf16: Optional[bool] = False,
    ):
        super().__init__(config_path, patch_size, stride)
        self.use_dtype = torch.float16 if use_fp16 else torch.float32
        self.use_dtype = (
            torch.bfloat16 if use_bf16 and is_bfloat16_available() else self.use_dtype
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/vae")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/vae")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        config_path = cached_path(config_path)

        patch_size = config.getoption("patch_size", 32)
        stride = config.getoption("stride", 16)
        use_fp16 = config.getoption("use_fp16", True)
        use_bf16 = config.getoption("use_bf16", False)

        inst = cls(
            config_path,
            patch_size=patch_size,
            stride=stride,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
        )

        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_infos, "vae", "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    def forward(self, pixel_values):
        with autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=self.use_dtype,
        ):
            loss = super().forward(pixel_values)
            return LossOutputs(loss=loss)
