# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.detr import DetrForDetection as _DetrForDetection
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import (
    detection_model_decorator,
    segmentation_model_decorator,
    DetectionOutputs,
    SegmentationOutputs,
    LossOutputs,
)
from unitorch.cli.models.detr import pretrained_detr_infos


@register_model("core/model/detection/detr", detection_model_decorator)
class DetrForDetection(_DetrForDetection):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = None,
    ):
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
        )

    @classmethod
    @add_default_section_for_init("core/model/detection/detr")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/detection/detr")
        pretrained_name = config.getoption("pretrained_name", "default-detr")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        num_classes = config.getoption("num_classes", None)

        inst = cls(
            config_path=config_path,
            num_classes=num_classes,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(self, images, bboxes, classes):
        outputs = super().forward(
            images=images,
            bboxes=bboxes,
            classes=classes,
        )
        return LossOutputs(loss=outputs)

    @add_default_section_for_function("core/model/detection/detr")
    def detect(
        self,
        images,
        norm_bboxes: bool = False,
    ):
        outputs = super().detect(
            images=images,
            norm_bboxes=norm_bboxes,
        )
        return DetectionOutputs(
            bboxes=outputs.bboxes,
            scores=outputs.scores,
            classes=outputs.classes,
        )
