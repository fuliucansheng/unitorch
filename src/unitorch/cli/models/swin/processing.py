# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.swin import SwinProcessor as _SwinProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.swin import pretrained_swin_infos


class SwinProcessor(_SwinProcessor):
    """Swin Transformer processor for image tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initialize the SwinProcessor.

        Args:
            vision_config_path (str): The path to the vision model configuration file.
        """
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/swin")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of SwinProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The SwinProcessor configuration.
        """
        config.set_default_section("core/process/swin")
        pretrained_name = config.getoption("pretrained_name", "default-swin")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_swin_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/swin/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Process image classification using SwinProcessor.

        Args:
            image (Union[Image.Image, str]): The input image or path to the image.

        Returns:
            TensorsInputs: The processed tensors inputs.
        """
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorsInputs(pixel_values=outputs.pixel_values)
