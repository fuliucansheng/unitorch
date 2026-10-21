# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from PIL import Image
from typing import List, Optional, Tuple, Union

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
        tokenizer_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        added_tokens: Optional[str] = None,
        tokenizer_class: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        image_size: Optional[Tuple[int, int]] = None,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ):
        super().__init__(
            tokenizer_path=tokenizer_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            added_tokens=added_tokens,
            tokenizer_class=tokenizer_class,
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
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name) or {}
        use_auth_token = config.getoption("use_auth_token", False)
        pretrained_model_path = config.getoption("pretrained_model_path", None)
        tokenizer_class = config.getoption("tokenizer_class", None)
        if tokenizer_class is None and pretrained_model_path is None:
            tokenizer_class = nested_dict_value(pretrained_infos, "text", "tokenizer_class")

        assets = {
            "tokenizer_path": ("tokenizer/tokenizer.json", ("text", "tokenizer")),
            "vocab_path": ("tokenizer/vocab.json", ("text", "vocab")),
            "merge_path": ("tokenizer/merges.txt", ("text", "merge")),
            "tokenizer_config": ("tokenizer/tokenizer_config.json", ("text", "tokenizer_config")),
            "special_tokens_map": ("tokenizer/special_tokens_map.json", ("text", "special_tokens_map")),
            "chat_template": ("tokenizer/chat_template.jinja", ("text", "chat_template")),
            "added_tokens": ("tokenizer/added_tokens.json", ("text", "added_tokens")),
            "vae_config_path": ("vae/config.json", ("vae", "config")),
        }
        paths = {}
        for option, (filename, keys) in assets.items():
            if pretrained_model_path is not None:
                default = os.path.join(pretrained_model_path, filename)
                if not os.path.isfile(default):
                    default = None
            else:
                default = nested_dict_value(pretrained_infos, *keys)
            path = pop_value(config.getoption(option, None), default, check_none=False)
            paths[option] = cached_path(path, use_auth_token=use_auth_token) if path else None

        return {
            **paths,
            "tokenizer_class": tokenizer_class,
            "max_seq_length": config.getoption("max_seq_length", 512),
            "image_size": config.getoption("image_size", None),
            "center_crop": config.getoption("center_crop", False),
            "random_flip": config.getoption("random_flip", False),
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
        refer_image: Union[Image.Image, str, List[Union[Image.Image, str]]],
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
        refer_image: Union[Image.Image, str, List[Union[Image.Image, str]]],
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
