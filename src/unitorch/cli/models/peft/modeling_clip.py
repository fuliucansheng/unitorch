# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import List, Optional, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.peft import (
    ClipLoraForMatching as _ClipLoraForMatching,
    ClipLoraForTextMatching as _ClipLoraForTextMatching,
)
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.clip import pretrained_clip_infos


@register_model("core/model/matching/peft/lora/clip")
class ClipLoraForMatching(_ClipLoraForMatching):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )

    @classmethod
    @config_defaults_init("core/model/matching/peft/lora/clip")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/matching/peft/lora/clip")
        pretrained_name = config.getoption("pretrained_name", "clip-vit-base-patch16")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        projection_dim = config.getoption("projection_dim", 512)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["q_proj", "v_proj"])

        inst = cls(
            config_path,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            weight_path.append(pretrained_lora_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)
