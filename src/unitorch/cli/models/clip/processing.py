# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.clip import ClipProcessor as _ClipProcessor
from unitorch.cli import (
    add_default_section_for_init,
    register_process,
)
from unitorch.cli import cached_path
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.clip import pretrained_clip_infos


class ClipProcessor(_ClipProcessor):
    """Processor for the CLIP model."""

    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/clip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/clip")
        pretrained_name = config.getoption("pretrained_name", "clip-vit-base-patch16")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/clip/classification")
    def _classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
            pixel_values=outputs.pixel_values,
        )

    @register_process("core/process/clip/text_classification")
    def _text_classification(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text_classification(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            position_ids=outputs.position_ids,
        )

    @register_process("core/process/clip/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().image_classification(image=image)
        return TensorInputs(pixel_values=outputs.pixel_values)
