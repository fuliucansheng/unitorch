# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.mask2former import (
    Mask2FormerForSegmentation as _Mask2FormerForSegmentation,
)
from unitorch.models.mask2former import Mask2FormerProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.mask2former import pretrained_mask2former_infos


class Mask2FormerForSegmentationPipeline(_Mask2FormerForSegmentation):
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
        self.processor = Mask2FormerProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/mask2former")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "mask2former-swin-tiny-ade-semantic",
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/mask2former")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_mask2former_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "weight"),
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
    @add_default_section_for_function("core/pipeline/mask2former")
    def __call__(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        width, height = image.size
        inputs = self.processor.classification(image)
        outputs = self.segment(inputs.pixel_values.unsqueeze(0).to(self._device))
        masks, classes = outputs.masks.sigmoid(), outputs.classes.argmax(dim=-1)
        # resize masks to the original image size with nn.functional.interpolate
        masks = torch.nn.functional.interpolate(
            masks,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.cpu().numpy()[0]
        classes = classes.cpu().numpy()[0]
        id2label = self.model.config.id2label
        labels = [id2label.get(int(cls), None) for cls in classes]
        return [
            (mask, label) for mask, label in zip(masks, labels) if label is not None
        ]
