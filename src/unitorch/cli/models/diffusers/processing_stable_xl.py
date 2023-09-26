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
from unitorch.cli.models.diffusers import pretrained_diffusers_infos


class StableXLProcessor(_StableXLProcessor):
    def __init__(
        self,
        vocab1_path: str,
        merge1_path: str,
        vocab2_path: str,
        merge2_path: str,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
    ):
        super().__init__(
            vocab1_path=vocab1_path,
            merge1_path=merge1_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/diffusers/stable_xl")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/diffusers/stable_xl")
        pretrained_name = config.getoption("pretrained_name", "stable-v2")
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

    @register_process("core/process/diffusers/stable_xl/text2image/inputs")
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

    @register_process("core/process/diffusers/stable_xl/image2image/inputs")
    def _image2image_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().image2image_inputs(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            negative_input_ids=outputs.negative_input_ids,
            negative_attention_mask=outputs.negative_attention_mask,
        )

    @register_process("core/process/diffusers/stable_xl/inpainting/inputs")
    def _inpainting_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().inpainting_inputs(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            pixel_masks=outputs.pixel_masks,
            negative_input_ids=outputs.negative_input_ids,
            negative_attention_mask=outputs.negative_attention_mask,
        )
