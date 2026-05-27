# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import cv2
from PIL import Image
from typing import List, Optional, Tuple, Union

from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import LucyProcessor as _LucyProcessor
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_process,
)
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.diffusers import pretrained_stable_infos


class LucyProcessor(_LucyProcessor):
    def __init__(
        self,
        vocab_path: str,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        position_start_id: Optional[int] = 0,
        video_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(
            vocab_path=vocab_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            video_size=video_size,
        )

    @classmethod
    @config_defaults_init("core/process/diffusion/lucy")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/lucy")
        pretrained_name = config.getoption("pretrained_name", "lucy-edit-v1.1-dev")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)
        use_auth_token = config.getoption("use_auth_token", False)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path, use_auth_token=use_auth_token)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path, use_auth_token=use_auth_token)

        return {
            "vocab_path": vocab_path,
            "vae_config_path": vae_config_path,
        }

    @register_process("core/process/diffusion/lucy/video_editing")
    def _video_editing(
        self,
        prompt: str,
        refer_video: Union[cv2.VideoCapture, str, List[Image.Image]],
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().video_editing(
            prompt=prompt,
            refer_video=refer_video,
            video=video,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            pixel_values=outputs.pixel_values,
            refer_pixel_values=outputs.refer_pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    @register_process("core/process/diffusion/lucy/video_editing/inputs")
    def _video_editing_inputs(
        self,
        prompt: str,
        refer_video: Union[cv2.VideoCapture, str, List[Image.Image]],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().video_editing_inputs(
            prompt=prompt,
            refer_video=refer_video,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            refer_pixel_values=outputs.refer_pixel_values,
            input_ids=outputs.input_ids,
            negative_input_ids=outputs.negative_input_ids,
            attention_mask=outputs.attention_mask,
            negative_attention_mask=outputs.negative_attention_mask,
        )
