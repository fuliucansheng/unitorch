# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.diffusers import ControlNetProcessor as _ControlNetProcessor
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


class ControlNetProcessor(_ControlNetProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
        pixel_mean: List[float] = [0.5],
        pixel_std: List[float] = [0.5],
        resize_shape: Optional[List[int]] = [224, 224],
        crop_shape: Optional[List[int]] = [224, 224],
    ):
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
        )
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

    @classmethod
    @add_default_section_for_init("core/process/diffusion/controlnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/controlnet")
        pretrained_name = config.getoption("pretrained_name", "controlnet-canny")
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

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
        }

    @register_process("core/process/diffusion/controlnet")
    def controlnet(
        self,
        prompt: str,
        image: Image.Image,
        condition_image: Image.Image,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().controlnet(
            prompt=prompt,
            image=image.convert("RGB"),
            condition_image=condition_image.convert("RGB"),
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            condition_pixel_values=outputs.condition_pixel_values,
        )

    @register_process("core/process/diffusion/controlnet/inputs")
    def controlnet_inputs(
        self,
        prompt: str,
        condition_image: Image.Image,
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().controlnet_inputs(
            prompt=prompt,
            condition_image=condition_image.convert("RGB"),
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
