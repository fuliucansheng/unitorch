# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import DreamboothXLProcessor as _DreamboothXLProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    TensorsInputs,
)
from unitorch.cli.models.diffusers import pretrained_diffusers_infos


class DreamboothXLProcessor(_DreamboothXLProcessor):
    def __init__(
        self,
        vocab1_path: str,
        merge1_path: str,
        vocab2_path: str,
        merge2_path: str,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        pad_token1: Optional[str] = "<|endoftext|>",
        pad_token2: Optional[str] = "!",
        image_size: Optional[int] = 512,
        center_crop: Optional[bool] = False,
    ):
        super().__init__(
            vocab1_path=vocab1_path,
            merge1_path=merge1_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            pad_token1=pad_token1,
            pad_token2=pad_token2,
            image_size=image_size,
            center_crop=center_crop,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusers/dreambooth_xl")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusers/dreambooth_xl")
        pretrained_name = config.getoption("pretrained_name", "stable-xl-base")
        pretrain_infos = nested_dict_value(pretrained_diffusers_infos, pretrained_name)

        vocab1_path = config.getoption("vocab1_path", None)
        vocab1_path = pop_value(
            vocab1_path,
            nested_dict_value(pretrain_infos, "text", "vocab"),
        )
        vocab1_path = cached_path(vocab1_path)

        merge1_path = config.getoption("merge1_path", None)
        merge1_path = pop_value(
            merge1_path,
            nested_dict_value(pretrain_infos, "text", "merge"),
        )
        merge1_path = cached_path(merge1_path)

        vocab2_path = config.getoption("vocab2_path", None)
        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrain_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        merge2_path = config.getoption("merge2_path", None)
        merge2_path = pop_value(
            merge2_path,
            nested_dict_value(pretrain_infos, "text2", "merge"),
        )
        merge2_path = cached_path(merge2_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        return {
            "vocab1_path": vocab1_path,
            "merge1_path": merge1_path,
            "vocab2_path": vocab2_path,
            "merge2_path": merge2_path,
            "vae_config_path": vae_config_path,
        }

    @register_process("core/process/diffusers/dreambooth_xl/text2image")
    def _text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        class_prompt: Optional[str] = None,
        class_image: Optional[Union[Image.Image, str]] = None,
        class_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image(
            prompt=prompt,
            image=image,
            prompt2=prompt2,
            class_prompt=class_prompt,
            class_image=class_image,
            class_prompt2=class_prompt2,
            max_seq_length=max_seq_length,
        )
        if class_prompt is None:
            return TensorsInputs(
                input_ids=outputs.input_ids,
                attention_mask=outputs.attention_mask,
                pixel_values=outputs.pixel_values,
                input2_ids=outputs.input2_ids,
                attention2_mask=outputs.attention2_mask,
                add_time_ids=outputs.add_time_ids,
            )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            input2_ids=outputs.input2_ids,
            attention2_mask=outputs.attention2_mask,
            class_input_ids=outputs.class_input_ids,
            class_attention_mask=outputs.class_attention_mask,
            class_pixel_values=outputs.class_pixel_values,
            class_input2_ids=outputs.class_input2_ids,
            class_attention2_mask=outputs.class_attention2_mask,
            add_time_ids=outputs.add_time_ids,
        )

    @register_process("core/process/diffusers/dreambooth_xl/text2image/inputs")
    def _text2image_inputs(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            negative_input_ids=outputs.negative_input_ids,
            negative_attention_mask=outputs.negative_attention_mask,
        )
