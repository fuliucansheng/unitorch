# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.sam import (
    SamForSegmentation as _SamForSegmentation,
    SamProcessor,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import SegmentationOutputs, LossOutputs
from unitorch.cli.models import segmentation_model_decorator
from unitorch.cli.models.sam import pretrained_sam_infos


@register_model("core/model/segmentation/sam", segmentation_model_decorator)
class SamForSegmentation(_SamForSegmentation):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        mask_threshold: Optional[float] = 0.0,
        pred_iou_thresh: Optional[float] = 0.88,
        stability_score_thresh: Optional[float] = 0.95,
        stability_score_offset: Optional[int] = 1,
        crops_nms_thresh: Optional[float] = 0.7,
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = SamProcessor(vision_config_path=vision_config_path)
        self.mask_threshold = mask_threshold
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.crops_nms_thresh = crops_nms_thresh

    @classmethod
    @add_default_section_for_init("core/model/segmentation/sam")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/segmentation/sam")
        pretrained_name = config.getoption("pretrained_name", "default-sam")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        inst = cls(
            config_path=config_path,
            vision_config_path=vision_config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("core/model/segmentation/sam")
    @torch.no_grad()
    def segment(
        self,
        pixel_values: torch.Tensor,
        input_points: torch.Tensor,
        input_boxes: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
        input_labels: Optional[torch.Tensor] = None,
    ):
        outputs = super().segment(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
        )
        processed_masks = self.processor.processing_masks(
            masks=outputs.masks,
            scores=outputs.scores,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            mask_threshold=self.mask_threshold,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            stability_score_offset=self.stability_score_offset,
            crops_nms_thresh=self.crops_nms_thresh,
        )
        return SegmentationOutputs(
            masks=processed_masks.masks,
        )
