# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models.detr import (
    DetrForDetection as _DetrForDetection,
)
from unitorch.models.detr import DetrProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.detr import pretrained_detr_infos


class DetrForDetectionPipeline(_DetrForDetection):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = DetrProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/detr")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/pipeline/detr")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "detr-resnet-50"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/detr")
    def __call__(
        self,
        image: Union[Image.Image, str],
        threshold: Optional[float] = 0.5,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.image(image)
        pixel_values = [inputs.image.to(self._device)]
        outputs = self.detect(
            pixel_values,
            norm_bboxes=True,
            threshold=threshold,
        )
        result_image = image.copy()
        bboxes = outputs["bboxes"][0].cpu().numpy()
        scores = outputs["scores"][0].cpu().numpy()
        classes = outputs["classes"][0].cpu().numpy()
        # draw boxes on the result image with class and score
        for bbox, score, classid in zip(bboxes, scores, classes):
            if score < threshold:
                continue
            bbox = bbox * np.array(
                [
                    result_image.width,
                    result_image.height,
                    result_image.width,
                    result_image.height,
                ]
            )
            bbox = list(map(int, bbox))
            result_image = result_image.copy()
            draw = ImageDraw.Draw(result_image)
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0] + 5, bbox[1] + 5), f"{classid}", fill="blue")
            draw.text((bbox[0] + 5, bbox[1] + 15), f"{score:.2f}", fill="green")

        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return result_image
