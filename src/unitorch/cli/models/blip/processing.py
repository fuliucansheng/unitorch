# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.blip import BlipProcessor as _BlipProcessor
from unitorch.cli import (
    config_defaults_init,
    register_process,
)
from unitorch.cli import cached_path
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.blip import pretrained_blip_infos


class BlipProcessor(_BlipProcessor):
    """Processor for the BLIP model."""

    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
        position_start_id: Optional[int] = 0,
    ):
        super().__init__(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            position_start_id=position_start_id,
        )

    @classmethod
    @config_defaults_init("core/process/blip")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/blip")
        pretrained_name = config.getoption(
            "pretrained_name", "blip-image-captioning-base"
        )

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_blip_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        return {
            "vocab_path": vocab_path,
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/blip/text_classification")
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

    @register_process("core/process/blip/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        outputs = super().image_classification(image=image)
        return TensorInputs(pixel_values=outputs.pixel_values)

    @register_process("core/process/blip/classification")
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

    @register_process("core/process/blip/generation")
    def _generation(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_gen_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        outputs = super().generation(
            text=text,
            image=image,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/process/blip/generation/labels")
    def _generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        outputs = super().generation_labels(
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.input_ids,
            masks=outputs.attention_mask,
        )

    @register_process("core/postprocess/blip/detokenize")
    def _detokenize(
        self,
        outputs: GenerationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.sequences.shape[0]

        decoded = super().detokenize(sequences=outputs.sequences)
        results["decoded"] = decoded
        return WriterOutputs(results)
