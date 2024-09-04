# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.segformer import SegformerProcessor as _SegformerProcessor
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
from unitorch.cli.models.segformer import pretrained_segformer_infos


class SegformerProcessor(_SegformerProcessor):
    """Segformer Transformer (Segformer) Processor for handling image processing tasks."""

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initialize SegformerProcessor.

        Args:
            vision_config_path (str): The path to the vision config file.
        """
        super().__init__(
            vision_config_path=vision_config_path,
        )

    @classmethod
    @add_default_section_for_init("core/process/segformer")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of SegformerProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The processed arguments for initializing the processor.
        """
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
