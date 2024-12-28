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
from unitorch.models.dpt import (
    DPTForDepthEstimation as _DPTForDepthEstimation,
    DPTProcessor,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.dpt import pretrained_dpt_infos


class DPTForDepthEstimationPipeline(_DPTForDepthEstimation):
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
        self.processor = DPTProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/dpt")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "dpt-large",
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/dpt")

        pretrained_name = config.getoption("pretrained_name", pretrained_name)
        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_dpt_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_dpt_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_dpt_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path=config_path,
            vision_config_path=vision_config_path,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/dpt")
    def __call__(
        self,
        image: Union[Image.Image, str],
    ):
        inputs = self.processor.classification(image)
        pixel_values = inputs.pixel_values.unsqueeze(0).to(self._device)
        outputs = self.forward(
            pixel_values,
        )
        masks = outputs[0].cpu().numpy().squeeze(0)
        result_image = Image.fromarray((masks * 255) / np.max(masks))
        result_image = result_image.resize(image.size)

        return result_image
