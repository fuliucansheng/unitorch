# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import cv2
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import WanProcessor as _WanProcessor
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


class WanProcessor(_WanProcessor):
    def __init__(
        self,
        vocab_path: str,
        vae_config_path: Optional[str] = None,
        image_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        video_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            vocab_path=vocab_path,
            vae_config_path=vae_config_path,
            image_config_path=image_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            video_size=video_size,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusion/wan")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/wan")
        pretrained_name = config.getoption("pretrained_name", "wan-v2.1-i2v-14b")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        image_config_path = config.getoption("image_config_path", None)
        image_config_path = pop_value(
            image_config_path,
            nested_dict_value(pretrained_infos, "image", "vision_config"),
            check_none=False,
        )
        if image_config_path is not None:
            image_config_path = cached_path(image_config_path)

        return {
            "vocab_path": vocab_path,
            "vae_config_path": vae_config_path,
            "image_config_path": image_config_path,
        }

    @register_process("core/process/diffusion/wan/text2video")
    def _text2video(
        self,
        prompt: str,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2video(
            prompt=prompt,
            video=video,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    @register_process("core/process/diffusion/wan/text2video/inputs")
    def _text2video_inputs(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2video_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            negative_input_ids=outputs.negative_input_ids,
            attention_mask=outputs.attention_mask,
            negative_attention_mask=outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusion/wan/image2video")
    def _image2video(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().image2video(
            prompt=prompt,
            image=image,
            video=video,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            condition_pixel_values=outputs.condition_pixel_values,
            vae_pixel_values=outputs.vae_pixel_values,
        )

    @register_process("core/process/diffusion/wan/image2video/inputs")
    def _image2video_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().image2video_inputs(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            condition_pixel_values=outputs.condition_pixel_values,
            vae_pixel_values=outputs.vae_pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            negative_input_ids=outputs.negative_input_ids,
            negative_attention_mask=outputs.negative_attention_mask,
        )
