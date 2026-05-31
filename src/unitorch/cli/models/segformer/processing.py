# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.segformer import SegformerProcessor as _SegformerProcessor
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.segformer import pretrained_segformer_infos


class SegformerProcessor(_SegformerProcessor):
    """Segformer processor for image segmentation tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @config_defaults_init("core/process/segformer")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/segformer")
        pretrained_name = config.getoption(
            "pretrained_name", "segformer-b2-human-parse-24"
        )
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_segformer_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/segformer/image_segmentation")
    def _segmentation_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorInputs(pixel_values=outputs.pixel_values)
