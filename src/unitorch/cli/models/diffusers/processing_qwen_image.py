# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import QWenImageProcessor as _QWenImageProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    TensorsInputs,
)
from unitorch.cli.models.diffusers import pretrained_stable_infos


class QWenImageProcessor(_QWenImageProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vision_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        image_size: Optional[Tuple[int, int]] = None,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            vae_config_path=vae_config_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            max_seq_length=max_seq_length,
            image_size=image_size,
            center_crop=center_crop,
            random_flip=random_flip,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusion/qwen_image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/qwen_image")
        pretrained_name = config.getoption("pretrained_name", "qwen-image")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(pretrained_infos, "text", "tokenizer_config"),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config) if tokenizer_config is not None else None
        )

        special_tokens_map = config.getoption("special_tokens_map", None)
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(pretrained_infos, "text", "special_tokens_map"),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_infos, "vision_config"),
            check_none=False,
        )
        if vision_config_path is not None:
            vision_config_path = cached_path(vision_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)
        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "vision_config_path": vision_config_path,
            "vae_config_path": vae_config_path,
            "tokenizer_config": tokenizer_config,
            "special_tokens_map": special_tokens_map,
        }

    @register_process("core/process/diffusion/qwen_image/text2image")
    def _text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image(
            prompt=prompt,
            image=image,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
        )

    @register_process("core/process/diffusion/qwen_image/text2image/inputs")
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

    @register_process("core/process/diffusion/qwen_image/editing/inputs")
    def _editing_inputs(
        self,
        prompt: str,
        refer_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().editing_inputs(
            prompt=prompt,
            refer_image=refer_image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )

        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            refer_pixel_values=outputs.refer_pixel_values,
            refer_image_grid_thw=outputs.refer_image_grid_thw,
            refer_vae_pixel_values=outputs.refer_vae_pixel_values,
            negative_input_ids=outputs.negative_input_ids,
            negative_attention_mask=outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusion/qwen_image/editing")
    def _editing(
        self,
        prompt: str,
        refer_image: Union[Image.Image, str],
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().editing(
            prompt=prompt,
            refer_image=refer_image,
            image=image,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            refer_image_grid_thw=outputs.refer_image_grid_thw,
            refer_pixel_values=outputs.refer_pixel_values,
            refer_vae_pixel_values=outputs.refer_vae_pixel_values,
        )
