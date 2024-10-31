# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import StableXLProcessor as _StableXLProcessor
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


class StableXLProcessor(_StableXLProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        merge2_path: str,
        vae_config_path: str,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        pad_token: Optional[str] = "<|endoftext|>",
        pad_token2: Optional[str] = "!",
        image_size: Optional[Tuple[int, int]] = None,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            pad_token=pad_token,
            pad_token2=pad_token2,
            image_size=image_size,
            center_crop=center_crop,
            random_flip=random_flip,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusion/stable_xl")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/stable_xl")
        pretrained_name = config.getoption("pretrained_name", "stable-xl-base")
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

        vocab2_path = config.getoption("vocab2_path", None)
        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrained_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        merge2_path = config.getoption("merge2_path", None)
        merge2_path = pop_value(
            merge2_path,
            nested_dict_value(pretrained_infos, "text2", "merge"),
        )
        merge2_path = cached_path(merge2_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "vocab2_path": vocab2_path,
            "merge2_path": merge2_path,
            "vae_config_path": vae_config_path,
        }

    @register_process("core/process/diffusion/stable_xl/text2image")
    def _text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image(
            prompt=prompt,
            image=image,
            prompt2=prompt2,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            input2_ids=outputs.input2_ids,
            attention2_mask=outputs.attention2_mask,
            add_time_ids=outputs.add_time_ids,
        )

    @register_process("core/process/diffusion/stable_xl/text2image/inputs")
    def _text2image_inputs(
        self,
        prompt: str,
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image_inputs(
            prompt=prompt,
            prompt2=prompt2,
            negative_prompt=negative_prompt,
            negative_prompt2=negative_prompt2,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            input2_ids=outputs.input2_ids,
            attention_mask=outputs.attention_mask,
            attention2_mask=outputs.attention2_mask,
            negative_input_ids=outputs.negative_input_ids,
            negative_attention_mask=outputs.negative_attention_mask,
            negative_input2_ids=outputs.negative_input2_ids,
            negative_attention2_mask=outputs.negative_attention2_mask,
        )

    @register_process("core/process/diffusion/stable_xl/image2image/inputs")
    def _image2image_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image_inputs(
            prompt=prompt,
            prompt2=prompt2,
            negative_prompt=negative_prompt,
            negative_prompt2=negative_prompt2,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().image2image_inputs(image=image)
        return TensorsInputs(
            pixel_values=image_outputs.pixel_values,
            input_ids=text_outputs.input_ids,
            input2_ids=text_outputs.input2_ids,
            attention_mask=text_outputs.attention_mask,
            attention2_mask=text_outputs.attention2_mask,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
            negative_input2_ids=text_outputs.negative_input2_ids,
            negative_attention2_mask=text_outputs.negative_attention2_mask,
        )

    @register_process("core/process/diffusion/stable_xl/inpainting")
    def _inpainting(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image(
            prompt=prompt,
            image=image,
            prompt2=prompt2,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().inpainting_inputs(
            image=image,
            mask_image=mask_image,
        )
        return TensorsInputs(
            input_ids=text_outputs.input_ids,
            input2_ids=text_outputs.input2_ids,
            attention_mask=text_outputs.attention_mask,
            attention2_mask=text_outputs.attention2_mask,
            add_time_ids=text_outputs.add_time_ids,
            pixel_values=image_outputs.pixel_values,
            pixel_masks=image_outputs.pixel_masks,
        )

    @register_process("core/process/diffusion/stable_xl/inpainting/inputs")
    def _inpainting_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        text_outputs = super().text2image_inputs(
            prompt=prompt,
            prompt2=prompt2,
            negative_prompt=negative_prompt,
            negative_prompt2=negative_prompt2,
            max_seq_length=max_seq_length,
        )
        image_outputs = super().inpainting_inputs(
            image=image,
            mask_image=mask_image,
        )
        return TensorsInputs(
            pixel_values=image_outputs.pixel_values,
            pixel_masks=image_outputs.pixel_masks,
            input_ids=text_outputs.input_ids,
            input2_ids=text_outputs.input2_ids,
            attention_mask=text_outputs.attention_mask,
            attention2_mask=text_outputs.attention2_mask,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_input2_ids=text_outputs.negative_input2_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
            negative_attention2_mask=text_outputs.negative_attention2_mask,
        )
