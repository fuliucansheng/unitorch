# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Optional, Tuple, Union

from unitorch.models.diffusers import Flux2Processor as _Flux2Processor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_process,
)
from unitorch.cli.models import TensorInputs
from unitorch.cli.models.diffusers import pretrained_stable_infos


class Flux2Processor(_Flux2Processor):
    def __init__(
        self,
        processor_name_or_path: str,
        processor_subfolder: Optional[str] = "tokenizer",
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        image_size: Optional[Tuple[int, int]] = None,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ):
        super().__init__(
            processor_name_or_path=processor_name_or_path,
            processor_subfolder=processor_subfolder,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            image_size=image_size,
            center_crop=center_crop,
            random_flip=random_flip,
            use_auth_token=use_auth_token,
        )

    @classmethod
    @config_defaults_init("core/process/diffusion/flux2")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/diffusion/flux2")
        pretrained_name = config.getoption("pretrained_name", "flux2-dev")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)
        use_auth_token = config.getoption("use_auth_token", False)

        processor_name_or_path = config.getoption("processor_name_or_path", None)
        processor_name_or_path = pop_value(
            processor_name_or_path,
            nested_dict_value(pretrained_infos, "processor", "name"),
        )

        processor_subfolder = config.getoption("processor_subfolder", None)
        processor_subfolder = pop_value(
            processor_subfolder,
            nested_dict_value(pretrained_infos, "processor", "subfolder"),
            check_none=False,
        )

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path, use_auth_token=use_auth_token)

        max_seq_length = config.getoption("max_seq_length", 512)
        image_size = config.getoption("image_size", None)
        center_crop = config.getoption("center_crop", False)
        random_flip = config.getoption("random_flip", False)

        return {
            "processor_name_or_path": processor_name_or_path,
            "processor_subfolder": processor_subfolder,
            "vae_config_path": vae_config_path,
            "max_seq_length": max_seq_length,
            "image_size": image_size,
            "center_crop": center_crop,
            "random_flip": random_flip,
            "use_auth_token": use_auth_token,
        }

    @register_process("core/process/diffusion/flux2/text2image")
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
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
        )

    @register_process("core/process/diffusion/flux2/text2image/inputs")
    def _text2image_inputs(
        self,
        prompt: str,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().text2image_inputs(
            prompt=prompt,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    @register_process("core/process/diffusion/flux2/editing")
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
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            refer_pixel_values=outputs.refer_pixel_values,
            pixel_values=outputs.pixel_values,
        )

    @register_process("core/process/diffusion/flux2/editing/inputs")
    def _editing_inputs(
        self,
        prompt: str,
        refer_image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().editing_inputs(
            prompt=prompt,
            refer_image=refer_image,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            refer_pixel_values=outputs.refer_pixel_values,
        )
