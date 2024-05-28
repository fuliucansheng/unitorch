# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import (
    MultiControlNetProcessor as _MultiControlNetProcessor,
)
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


class MultiControlNetProcessor(_MultiControlNetProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vae_config_path: str,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        pad_token: Optional[str] = "<|endoftext|>",
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            pad_token=pad_token,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusers/multicontrolnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusers/multicontrolnet")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-v1.5-multicontrolnet-canny-depth"
        )
        pretrain_infos = nested_dict_value(pretrained_diffusers_infos, pretrained_name)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrain_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrain_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "vae_config_path": vae_config_path,
        }

    @register_process("core/process/diffusers/multicontrolnet/text2image")
    def _text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        condition_images: List[Union[Image.Image, str]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image(
            prompt=prompt,
            image=image,
            condition_images=condition_images,
            max_seq_length=max_seq_length,
        )

        return TensorsInputs(
            input_ids=outputs.input_ids,
            pixel_values=outputs.pixel_values,
            condition_pixel_values=outputs.condition_pixel_values,
            attention_mask=outputs.attention_mask,
        )

    @register_process("core/process/diffusers/multicontrolnet/text2image/inputs")
    def _text2image_inputs(
        self,
        prompt: str,
        condition_images: List[Union[Image.Image, str]],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image_inputs(
            prompt=prompt,
            condition_images=condition_images,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            negative_input_ids=outputs.negative_input_ids,
            condition_pixel_values=outputs.condition_pixel_values,
            attention_mask=outputs.attention_mask,
            negative_attention_mask=outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusers/multicontrolnet/image2image/inputs")
    def _image2image_inputs(
        self,
        prompt: str,
        condition_images: List[Union[Image.Image, str]],
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().image2image_inputs(
            prompt=prompt,
            condition_images=condition_images,
            image=image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            negative_input_ids=outputs.negative_input_ids,
            pixel_values=outputs.pixel_values,
            condition_pixel_values=outputs.condition_pixel_values,
            attention_mask=outputs.attention_mask,
            negative_attention_mask=outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusers/multicontrolnet/inpainting/inputs")
    def _inpainting_inputs(
        prompt: str,
        condition_images: List[Union[Image.Image, str]],
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().inpainting_inputs(
            prompt=prompt,
            condition_images=condition_images,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            negative_input_ids=outputs.negative_input_ids,
            pixel_values=outputs.pixel_values,
            condition_pixel_values=outputs.condition_pixel_values,
            attention_mask=outputs.attention_mask,
            negative_attention_mask=outputs.negative_attention_mask,
        )
