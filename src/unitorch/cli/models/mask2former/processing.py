# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.mask2former import Mask2FormerProcessor as _Mask2FormerProcessor
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.mask2former import pretrained_mask2former_infos


class Mask2FormerProcessor(_Mask2FormerProcessor):
    """Mask2Former processor for image segmentation tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @config_defaults_init("core/process/mask2former")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/mask2former")
        pretrained_name = config.getoption(
            "pretrained_name", "mask2former-swin-tiny-ade-semantic"
        )
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_mask2former_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/mask2former/image_segmentation")
    def _segmentation_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorInputs(pixel_values=outputs.pixel_values)
