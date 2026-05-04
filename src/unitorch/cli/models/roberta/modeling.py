# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.roberta import (
    RobertaForClassification as _RobertaForClassification,
)
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.roberta import pretrained_roberta_infos


@register_model("core/model/classification/roberta")
class RobertaForClassification(_RobertaForClassification):
    """Roberta model for classification tasks."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @config_defaults_init("core/model/classification/roberta")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/classification/roberta")
        pretrained_name = config.getoption("pretrained_name", "roberta-base")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_roberta_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(config_path, num_classes, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_roberta_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)
