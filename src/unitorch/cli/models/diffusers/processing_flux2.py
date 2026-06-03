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
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)
        use_auth_token = config.getoption("use_auth_token", False)

        tokenizer_class = config.getoption("tokenizer_class", None)
        tokenizer_class = pop_value(
            tokenizer_class,
            nested_dict_value(pretrained_infos, "text", "tokenizer_class"),
            check_none=False,
        )

        tokenizer_path = config.getoption("tokenizer_path", None)
        tokenizer_path = pop_value(
            tokenizer_path,
            nested_dict_value(pretrained_infos, "text", "tokenizer"),
            check_none=False,
        )
        tokenizer_path = (
            cached_path(tokenizer_path, use_auth_token=use_auth_token)
            if tokenizer_path is not None
            else None
        )

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
            check_none=False,
        )
        vocab_path = (
            cached_path(vocab_path, use_auth_token=use_auth_token)
            if vocab_path is not None
            else None
        )

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
            check_none=False,
        )
        merge_path = (
            cached_path(merge_path, use_auth_token=use_auth_token)
            if merge_path is not None
            else None
        )

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(pretrained_infos, "text", "tokenizer_config"),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config, use_auth_token=use_auth_token)
            if tokenizer_config is not None
            else None
        )

        special_tokens_map = config.getoption("special_tokens_map", None)
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(pretrained_infos, "text", "special_tokens_map"),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map, use_auth_token=use_auth_token)
            if special_tokens_map is not None
            else None
        )

        chat_template = config.getoption("chat_template", None)
        chat_template = pop_value(
            chat_template,
            nested_dict_value(pretrained_infos, "text", "chat_template"),
            check_none=False,
        )
        chat_template = (
            cached_path(chat_template, use_auth_token=use_auth_token)
            if chat_template is not None
            else None
        )

        added_tokens = config.getoption("added_tokens", None)
        added_tokens = pop_value(
            added_tokens,
            nested_dict_value(pretrained_infos, "text", "added_tokens"),
            check_none=False,
        )
        added_tokens = (
            cached_path(added_tokens, use_auth_token=use_auth_token)
            if added_tokens is not None
            else None
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
            "tokenizer_path": tokenizer_path,
            "vocab_path": vocab_path,
            "merge_path": merge_path,
            "tokenizer_config": tokenizer_config,
            "special_tokens_map": special_tokens_map,
            "chat_template": chat_template,
            "added_tokens": added_tokens,
            "tokenizer_class": tokenizer_class,
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
