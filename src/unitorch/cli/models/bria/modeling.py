# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from unitorch.models.bria import BRIAForSegmentation as _BRIAForSegmentation
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


@register_model("core/model/segmentation/bria", segmentation_model_decorator)
class BRIAForSegmentation(_BRIAForSegmentation):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__(in_ch, out_ch)

    @classmethod
    @add_default_section_for_init("core/model/segmentation/bria")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/segmentation/bria")
        in_ch = config.getoption("in_ch", 3)
        out_ch = config.getoption("out_ch", 1)

        inst = cls(
            in_ch=in_ch,
            out_ch=out_ch,
        )
        weight_path = config.getoption("pretrained_weight_path", None)
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(self, images):
        outputs = super().forward(images)
        return SegmentationOutputs(
            masks=outputs,
        )

    def segment(self, images, sizes: Optional[List[Tuple[int, int]]] = None):
        outputs = super().forward(images)
        if sizes is None:
            return SegmentationOutputs(
                masks=outputs,
            )
        masks = [
            F.interpolate(mask.unsqueeze(0), size=list(size), mode="bilinear").squeeze(
                0
            )
            for mask, size in zip(outputs, sizes)
        ]
        masks = [m.permute(1, 2, 0) for m in masks]
        masks = [(m - m.min()) / (m.max() - m.min()) for m in masks]
        return SegmentationOutputs(
            masks=masks,
        )
