# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.mask2former import (
    Mask2FormerForSegmentation as _Mask2FormerForSegmentation,
    Mask2FormerProcessor,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import SegmentationOutputs, LossOutputs
from unitorch.cli.models import segmentation_model_decorator
from unitorch.cli.models.mask2former import pretrained_mask2former_infos


@register_model("core/model/segmentation/mask2former", segmentation_model_decorator)
class Mask2FormerForSegmentation(_Mask2FormerForSegmentation):
    def __init__(
        self,
        config_path: str,
    ):
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @add_default_section_for_init("core/model/segmentation/mask2former")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/segmentation/mask2former")
        pretrained_name = config.getoption(
            "pretrained_name", "mask2former-swin-tiny-ade-semantic"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        inst = cls(
            config_path=config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("core/model/segmentation/mask2former")
    @torch.no_grad()
    def segment(
        self,
        pixel_values: torch.Tensor,
    ):
        outputs = super().segment(
            pixel_values=pixel_values,
        )
        return SegmentationOutputs(
            masks=list(outputs.masks),
            classes=list(outputs.classes),
        )
