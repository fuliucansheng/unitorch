# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.mask2former import Mask2FormerProcessor as _Mask2FormerProcessor
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
from unitorch.cli.models.mask2former import pretrained_mask2former_infos


class Mask2FormerProcessor(_Mask2FormerProcessor):
    """Mask2Former Transformer (Mask2Former) Processor for handling image processing tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initialize Mask2FormerProcessor.

        Args:
            vision_config_path (str): The path to the vision config file.
        """
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/mask2former")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Mask2FormerProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The processed arguments for initializing the processor.
        """
        config.set_default_section("core/process/mask2former")
        pretrained_name = config.getoption("pretrained_name", "default-mask2former")
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
        """
        Process an image for image classification tasks.

        Args:
            image (Union[Image.Image, str]): The input image or path to the image.

        Returns:
            TensorsInputs: The processed input tensors.
        """

        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().classification(image=image)
        return TensorsInputs(pixel_values=outputs.pixel_values)
