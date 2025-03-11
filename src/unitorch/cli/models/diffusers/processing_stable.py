# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import StableProcessor as _StableProcessor
from unitorch.models.diffusers import StableVideoProcessor as _StableVideoProcessor
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


class StableProcessor(_StableProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vae_config_path: str,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        pad_token: Optional[str] = "<|endoftext|>",
        image_size: Optional[Tuple[int, int]] = None,
        low_res_image_size: Optional[Tuple[int, int]] = None,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            pad_token=pad_token,
            image_size=image_size,
            low_res_image_size=low_res_image_size,
            center_crop=center_crop,
            random_flip=random_flip,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusion/stable")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/stable")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
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

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "vae_config_path": vae_config_path,
        }

    @register_process("core/process/diffusion/stable/text2image")
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

    @register_process("core/process/diffusion/stable/text2image/inputs")
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

    @register_process("core/process/diffusion/stable/image2image/inputs")
    def _image2image_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().image2image_inputs(image=image)
        return TensorsInputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            pixel_values=image_outputs.pixel_values,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusion/stable/vae")
    def _vae(
        self,
        image: Union[Image.Image, str],
    ):
        image_outputs = super().image2image_inputs(image=image)
        return TensorsInputs(
            pixel_values=image_outputs.pixel_values,
        )

    @register_process("core/process/diffusion/stable/inpainting")
    def _inpainting(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().inpainting_inputs(
            image=image,
            mask_image=mask_image,
        )
        return TensorsInputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            pixel_values=image_outputs.pixel_values,
            pixel_masks=image_outputs.pixel_masks,
        )

    @register_process("core/process/diffusion/stable/inpainting/inputs")
    def _inpainting_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().inpainting_inputs(
            image=image,
            mask_image=mask_image,
        )
        return TensorsInputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            pixel_values=image_outputs.pixel_values,
            pixel_masks=image_outputs.pixel_masks,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusion/stable/resolution")
    def _resolution(
        self,
        prompt: str,
        low_res_image: Union[Image.Image, str],
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image(
            prompt=prompt,
            image=image,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().resolution_inputs(image=low_res_image)
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            low_res_pixel_values=image_outputs.pixel_values,
        )

    @register_process("core/process/diffusion/stable/resolution/inputs")
    def _resolution_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().resolution_inputs(image=image)
        return TensorsInputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            pixel_values=image_outputs.pixel_values,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
        )
