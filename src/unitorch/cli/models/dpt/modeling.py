# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.dpt import (
    DPTForDepthEstimation as _DPTForDepthEstimation,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import SegmentationOutputs, LossOutputs
from unitorch.cli.models import segmentation_model_decorator
from unitorch.cli.models.dpt import pretrained_dpt_infos


@register_model("core/model/dpt", segmentation_model_decorator)
class DPTForDepthEstimation(_DPTForDepthEstimation):
    def __init__(
        self,
        config_path: str,
    ):
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @add_default_section_for_init("core/model/dpt")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/dpt")
        pretrained_name = config.getoption("pretrained_name", "dpt-large")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_dpt_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        inst = cls(
            config_path=config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_dpt_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("core/model/dpt")
    @torch.no_grad()
    def segment(
        self,
        pixel_values: torch.Tensor,
    ):
        outputs = super().forward(
            pixel_values=pixel_values,
        )
        max_values = torch.amax(outputs.reshape(len(outputs), -1), dim=1)
        return SegmentationOutputs(
            masks=list([out / value for out, value in zip(outputs, max_values)]),
        )
