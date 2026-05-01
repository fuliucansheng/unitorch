# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.vit import ViTProcessor as _ViTProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    register_process,
)
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.vit import pretrained_vit_infos


class ViTProcessor(_ViTProcessor):
    """Vision Transformer (ViT) processor for image tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/vit")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/vit")
        pretrained_name = config.getoption(
            "pretrained_name", "vit-base-patch16-224-in21k"
        )
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_vit_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/vit/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorInputs(pixel_values=outputs.pixel_values)
