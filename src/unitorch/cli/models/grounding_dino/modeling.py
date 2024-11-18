# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.grounding_dino import (
    GroundingDinoForDetection as _GroundingDinoForDetection,
)
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
from unitorch.cli.models.grounding_dino import pretrained_grounding_dino_infos


@register_model("core/model/detection/grounding_dino", detection_model_decorator)
class GroundingDinoForDetection(_GroundingDinoForDetection):
    def __init__(
        self,
        config_path: str,
    ):
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @add_default_section_for_init("core/model/detection/grounding_dino")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/detection/grounding_dino")
        pretrained_name = config.getoption("pretrained_name", "grounding-dino-tiny")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "config"
            ),
        )

        config_path = cached_path(config_path)

        inst = cls(
            config_path=config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "weight"
            ),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self, pixel_values, input_ids, attention_mask, token_type_ids, bboxes, classes
    ):
        outputs = super().forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bboxes=bboxes,
            classes=classes,
        )
        return LossOutputs(loss=outputs)

    @add_default_section_for_function("core/model/detection/grounding_dino")
    def detect(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        token_type_ids,
        norm_bboxes: bool = False,
        text_threshold: Optional[float] = 0.25,
        box_threshold: Optional[float] = 0.25,
    ):
        outputs = super().detect(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            norm_bboxes=norm_bboxes,
            text_threshold=text_threshold,
            box_threshold=box_threshold,
        )
        classes = [c.max(dim=-1)[1] for c in outputs.classes]
        return DetectionOutputs(
            bboxes=outputs.bboxes,
            scores=outputs.scores,
            classes=classes,
        )
