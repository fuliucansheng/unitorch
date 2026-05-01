# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.dinov2 import (
    DinoV2ForImageClassification as _DinoV2ForImageClassification,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.dinov2 import pretrained_dinov2_infos


@register_model("core/model/classification/dinov2")
class DinoV2ForImageClassification(_DinoV2ForImageClassification):
    """DINOv2 model for image classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/dinov2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/dinov2")
        pretrained_name = config.getoption("pretrained_name", "dinov2-base")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_dinov2_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(
            config_path=config_path,
            num_classes=num_classes,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_dinov2_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        outputs = super().forward(pixel_values=pixel_values)
        return ClassificationOutputs(outputs=outputs)
