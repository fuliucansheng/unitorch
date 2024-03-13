# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from PIL import Image
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from unitorch.cli.models.segmentation_utils import SegmentationTargets
from unitorch.models.bria import BRIAProcessor as _BRIAProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import TensorsInputs


class BRIAProcessor(_BRIAProcessor):
    def __init__(
        self,
        image_size: Optional[int] = 1024,
    ):
        super().__init__(image_size=image_size)

    @classmethod
    @add_default_section_for_init("core/process/bria")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/bria")
        image_size = config.getoption("image_size", 1024)

        return {
            "image_size": image_size,
        }

    @register_process("core/process/bria/segmentation/inputs")
    def _segmentation_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        inputs = super().segmentation_inputs(image=image)
        return TensorsInputs(images=inputs.image, sizes=inputs.sizes)

    @register_process("core/process/bria/segmentation")
    def _segmentation(
        self,
        image: Union[Image.Image, str],
        mask: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(mask, str):
            mask = Image.open(mask)
        inputs = super().segmentation_inputs(image=image)
        labels = super().segmentation_labels(image=mask)
        return TensorsInputs(images=inputs.image), SegmentationTargets(
            targets=labels.image
        )
