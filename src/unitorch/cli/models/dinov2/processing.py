# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.dinov2 import DinoV2Processor as _DinoV2Processor
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_process,
)
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.dinov2 import pretrained_dinov2_infos


class DinoV2Processor(_DinoV2Processor):
    """DINOv2 processor for image tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @config_defaults_init("core/process/dinov2")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/dinov2")
        pretrained_name = config.getoption("pretrained_name", "dinov2-base")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_dinov2_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/dinov2/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorInputs(pixel_values=outputs.pixel_values)
