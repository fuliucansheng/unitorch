# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.segformer import (
    SegformerForSegmentation as _SegformerForSegmentation,
    SegformerProcessor,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import SegmentationOutputs, LossOutputs
from unitorch.cli.models import segmentation_model_decorator
from unitorch.cli.models.segformer import pretrained_segformer_infos


@register_model("core/model/segmentation/segformer", segmentation_model_decorator)
class SegformerForSegmentation(_SegformerForSegmentation):
    def __init__(
        self,
        config_path: str,
    ):
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @add_default_section_for_init("core/model/segmentation/segformer")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/segmentation/segformer")
        pretrained_name = config.getoption(
            "pretrained_name", "segformer-b2-human-parse-24"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_segformer_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        inst = cls(
            config_path=config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_segformer_infos, pretrained_name, "weight"),
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

    @add_default_section_for_function("core/model/segmentation/segformer")
    @torch.no_grad()
    def segment(
        self,
        pixel_values: torch.Tensor,
    ):
        outputs = super().forward(
            pixel_values=pixel_values,
        )
        # batch, width, height, num_classes -> batch, num_classes, width, height + batch, num_classes
        batch = outputs.logits.shape[0]
        num_classes = outputs.logits.shape[-1]

        masks = torch.softmax(outputs.logits, dim=1)
        # set the logit of not the highest class to 0
        masks = masks * (masks == masks.max(dim=1, keepdim=True).values).float()
        classes = (
            torch.arange(num_classes, device=masks.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )

        return SegmentationOutputs(
            masks=masks,
            classes=classes,
        )
