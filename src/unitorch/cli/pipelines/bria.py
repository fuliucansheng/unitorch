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
        in_channels: Optional[int] = 3,
        out_channels: Optional[int] = 1,
        image_size: Optional[int] = 1024,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            in_ch=in_channels,
            out_ch=out_channels,
        )
        self.processor = BRIAProcessor(
            image_size=image_size,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/bria")
    def from_core_configure(
        cls,
        config,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/bria")

        in_channels = config.getoption("in_channels", 3)
        out_channels = config.getoption("out_channels", 1)
        image_size = config.getoption("image_size", 1024)
        device = config.getoption("device", device)
        weight_path = config.getoption("pretrained_weight_path", pretrained_weight_path)

        inst = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            weight_path=weight_path,
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
        inputs = self.processor.segmentation_inputs(image)
        pixel_values = inputs.image.unsqueeze(0).to(self._device)
        outputs = self.forward(
            pixel_values,
        )
        masks = [
            (mask.squeeze(0).cpu().numpy() > threshold).astype(np.uint8)
            for mask in outputs
        ][0]
        result_image = Image.fromarray(masks * 255)
        result_image = result_image.resize(image.size)

        return result_image
