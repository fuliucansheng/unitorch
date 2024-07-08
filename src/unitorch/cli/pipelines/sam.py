# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.sam import (
    SamForSegmentation as _SamForSegmentation,
)
from unitorch.models.sam import SamProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.sam import pretrained_sam_infos


class SamForSegmentationPipeline(_SamForSegmentation):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = SamProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/sam")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "default-sam",
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/sam")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/sam")
    def __call__(
        self,
        image: Union[Image.Image, str],
        points: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        mask_threshold: Optional[float] = 0.1,
    ):
        inputs = self.processor.vision_processor(image)
        pixel_values, original_sizes, reshaped_input_sizes = (
            inputs["pixel_values"],
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        pixel_values = [torch.from_numpy(pixel_value) for pixel_value in pixel_values]
        pixel_values = torch.stack(pixel_values).to(self._device)
        input_points = (
            torch.Tensor([[points]]).to(self._device) if points is not None else None
        )
        input_boxes = (
            torch.Tensor([boxes]).to(self._device) if boxes is not None else None
        )
        if input_points is not None:
            input_points[:, :, :, 1] = (
                input_points[:, :, :, 1]
                / original_sizes[0][0]
                * reshaped_input_sizes[0][0]
            )
            input_points[:, :, :, 0] = (
                input_points[:, :, :, 0]
                / original_sizes[0][1]
                * reshaped_input_sizes[0][1]
            )
        outputs = self.segment(
            pixel_values,
            input_points=input_points,
            input_boxes=input_boxes,
        )
        processed_masks = self.processor.vision_processor.post_process_masks(
            outputs.masks,
            original_sizes,
            reshaped_input_sizes,
            mask_threshold=mask_threshold,
            binarize=True,
        )[0]
        if len(processed_masks) == 0:
            return None
        first_mask = processed_masks[0, 0].permute(0, 1)
        first_mask = first_mask.cpu().to(torch.uint8) * 255
        mask_image = Image.fromarray(np.array(first_mask))
        return mask_image
