# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.beit import BeitProcessor as _BeitProcessor
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
from unitorch.cli.models.beit import pretrained_beit_infos


class BeitProcessor(_BeitProcessor):
    """Class for processing images using the Beit model."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initialize BeitProcessor.

        Args:
            vision_config_path (str): The path to the vision configuration file.
        """
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/beit")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BeitProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            BeitProcessor: An instance of BeitProcessor.
        """
        config.set_default_section("core/process/beit")
        pretrained_name = config.getoption("pretrained_name", "default-beit")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_beit_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/beit/classification")
    def _classification(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Perform image classification using BeitProcessor.

        Args:
            image (Union[Image.Image, str]): The input image.

        Returns:
            TensorsInputs: The input tensors for the model.
        """
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorsInputs(pixel_values=outputs.pixel_values)
