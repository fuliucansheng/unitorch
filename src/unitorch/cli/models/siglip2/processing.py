# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.siglip2 import Siglip2Processor as _Siglip2Processor
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import cached_path
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.siglip2 import pretrained_siglip2_infos


class Siglip2Processor(_Siglip2Processor):
    """Processor for the CLIP model."""

    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        """
        Initialize the Siglip2Processor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the BPE merge file.
            vision_config_path (str): The path to the vision configuration file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            position_start_id (int, optional): The position start ID. Defaults to 0.
        """
        super().__init__(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/siglip2")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Siglip2Processor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the processor's initialization arguments.
        """
        config.set_default_section("core/process/siglip2")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip2-base-patch16-224"
        )
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_siglip2_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vocab_path": vocab_path,
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/siglip2/classification")
    def _classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        """
        Process text and image inputs for classification.

        Args:
            text (str): The input text.
            image (Union[Image.Image, str]): The input image.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The processed inputs as tensors.
        """
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
            pixel_values=outputs.pixel_values,
            pixel_masks=outputs.pixel_masks,
            spatial_shapes=outputs.spatial_shapes,
        )

    @register_process("core/process/siglip2/text_classification")
    def _text_classification(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Process text inputs for text classification.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The processed inputs as tensors.
        """
        outputs = super().text_classification(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
        )

    @register_process("core/process/siglip2/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Process image inputs for image classification.

        Args:
            image (Union[Image.Image, str]): The input image.

        Returns:
            TensorsInputs: The processed inputs as tensors.
        """
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().image_classification(image=image)
        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            pixel_masks=outputs.pixel_masks,
            spatial_shapes=outputs.spatial_shapes,
        )
