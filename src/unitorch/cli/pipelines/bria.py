# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
import numpy as np
import pandas as pd
import hashlib
from PIL import Image, ImageDraw
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bria import (
    BRIAProcessor,
    BRIAForSegmentation as _BRIAForSegmentation,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script


class BRIAForSegmentationPipeline(_BRIAForSegmentation):
    def __init__(
        self,
        image_size: Optional[int] = 1024,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__()
        self.processor = BRIAProcessor(
            image_size=image_size,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/bria")
    def from_core_configure(
        cls,
        config,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/pipeline/bria")

        image_size = config.getoption("image_size", 1024)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )

        inst = cls(
            image_size=image_size,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/bria")
    def __call__(
        self,
        image: Union[Image.Image, str],
        threshold: Optional[float] = 0.5,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)

        inputs = self.processor.segmentation_inputs(image)
        pixel_values = inputs.image.unsqueeze(0).to(self._device)
        outputs = self.forward(
            pixel_values,
        ).logits
        masks = [
            (mask.squeeze(0).cpu().numpy() > threshold).astype(np.uint8)
            for mask in outputs
        ][0]
        result_image = Image.fromarray(masks * 255)
        result_image = result_image.resize(image.size, resample=Image.LANCZOS)

        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()

        return result_image
