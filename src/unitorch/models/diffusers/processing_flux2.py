# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from transformers import LlamaTokenizerFast, Qwen2Tokenizer

from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from unitorch.models import GenericOutputs
from unitorch.utils import get_added_token, pop_value, read_file, read_json_file


def _load_chat_template(chat_template: Optional[str]) -> Optional[str]:
    if chat_template is None:
        return None
    if chat_template.endswith(".json"):
        data = read_json_file(chat_template)
        if isinstance(data, dict) and "chat_template" in data:
            return data["chat_template"]
        if isinstance(data, str):
            return data
    return read_file(chat_template)


def _get_special_token(spec):
    if isinstance(spec, list):
        return [get_added_token(v) if isinstance(v, (str, dict)) else v for v in spec]
    if isinstance(spec, (str, dict)):
        return get_added_token(spec)
    return spec


def _build_flux2_tokenizer(
    tokenizer_class: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    merge_path: Optional[str] = None,
    tokenizer_config: Optional[str] = None,
    special_tokens_map: Optional[str] = None,
    chat_template: Optional[str] = None,
    added_tokens: Optional[str] = None,
):
    tokenizer_config = read_json_file(tokenizer_config) if tokenizer_config else {}
    special_tokens_map = (
        read_json_file(special_tokens_map) if special_tokens_map else {}
    )
    added_tokens = read_json_file(added_tokens) if added_tokens else {}

    added_tokens_decoder = tokenizer_config.pop("added_tokens_decoder", {})
    tokenizer_config = {
        k: (
            get_added_token(v)
            if isinstance(v, dict) and v.get("__type") == "AddedToken"
            else v
        )
        for k, v in tokenizer_config.items()
    }
    tokenizer_class = tokenizer_class or tokenizer_config.get("tokenizer_class")

    if tokenizer_class in (None, "LlamaTokenizerFast"):
        if tokenizer_path is None:
            raise ValueError("Flux2Processor requires tokenizer_path for LlamaTokenizerFast.")
        tokenizer = LlamaTokenizerFast(
            tokenizer_file=tokenizer_path,
            **tokenizer_config,
        )
    elif tokenizer_class in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
        if vocab_path is None or merge_path is None:
            raise ValueError(
                "Flux2Processor requires vocab_path and merge_path for Qwen2Tokenizer."
            )
        tokenizer = Qwen2Tokenizer(
            vocab=vocab_path,
            merges=merge_path,
            **tokenizer_config,
        )
    else:
        raise ValueError(f"Unsupported FLUX.2 tokenizer class: {tokenizer_class}")

    for idx, spec in added_tokens_decoder.items():
        idx = int(idx)
        token = spec["content"]
        tokenizer.added_tokens_decoder[idx] = get_added_token(spec)
        tokenizer.added_tokens_encoder[token] = idx

    for token, idx in added_tokens.items():
        idx = int(idx)
        if idx in tokenizer.added_tokens_decoder:
            continue
        tokenizer.added_tokens_decoder[idx] = get_added_token(token)
        tokenizer.added_tokens_encoder[token] = idx

    special_tokens = {}
    for name, spec in special_tokens_map.items():
        special_tokens[name] = _get_special_token(spec)
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    template = _load_chat_template(chat_template)
    if template:
        tokenizer.chat_template = template

    if tokenizer.cls_token is None and tokenizer.bos_token is not None:
        tokenizer.cls_token = tokenizer.bos_token
    if tokenizer.sep_token is None and tokenizer.eos_token is not None:
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.cls_token_id is None and tokenizer.bos_token_id is not None:
        tokenizer.cls_token_id = tokenizer.bos_token_id
    if tokenizer.sep_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.sep_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.unk_token_id
        )
    return tokenizer


class Flux2Processor:
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
        self.tokenizer = _build_flux2_tokenizer(
            tokenizer_class=tokenizer_class,
            tokenizer_path=tokenizer_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            added_tokens=added_tokens,
        )
        self.processor = self.tokenizer
        self.max_seq_length = max_seq_length

        if image_size is not None:
            self.image_size = (
                image_size
                if isinstance(image_size, tuple)
                else (image_size, image_size)
            )
        else:
            self.image_size = None

        if self.image_size is not None:
            self.vision_processor = Compose(
                [
                    Resize((self.image_size[1], self.image_size[0])),
                    (
                        CenterCrop((self.image_size[1], self.image_size[0]))
                        if center_crop
                        else RandomCrop((self.image_size[1], self.image_size[0]))
                    ),
                    RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
                    ToTensor(),
                    Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.vision_processor = Compose(
                [
                    RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
                    ToTensor(),
                    Normalize([0.5], [0.5]),
                ]
            )

        if vae_config_path is not None:
            with open(vae_config_path) as f:
                vae_config_dict = json.load(f)
            vae_scale_factor = 2 ** (
                len(vae_config_dict.get("block_out_channels", [])) - 1
            )
        else:
            vae_scale_factor = 8

        self.vae_image_processor = Flux2ImageProcessor(
            vae_scale_factor=vae_scale_factor * 2
        )

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        return image.convert("RGB")

    def _tokenize_prompt(
        self,
        prompt: str,
        max_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        messages = [
            {"role": "system", "content": str(SYSTEM_MESSAGE)},
            {"role": "user", "content": str(prompt).replace("[IMG]", "")},
        ]
        outputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        return GenericOutputs(
            input_ids=outputs["input_ids"][0].long(),
            attention_mask=outputs["attention_mask"][0].long(),
        )

    def _process_reference_image(
        self,
        image: Union[Image.Image, str],
    ):
        image = self._load_image(image)
        if self.image_size is not None:
            height, width = self.image_size[1], self.image_size[0]
            pixel_values = self.vae_image_processor.preprocess(
                image,
                height=height,
                width=width,
                resize_mode="crop",
            )[0]
        else:
            pixel_values = self.vae_image_processor.preprocess(image)[0]
        return pixel_values

    def text2image_inputs(
        self,
        prompt: str,
        max_seq_length: Optional[int] = None,
    ):
        return self._tokenize_prompt(
            prompt=prompt,
            max_seq_length=max_seq_length,
        )

    def text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.text2image_inputs(
            prompt=prompt,
            max_seq_length=max_seq_length,
        )
        image = self._load_image(image)
        pixel_values = self.vision_processor(image)

        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            pixel_values=pixel_values,
        )

    def editing_inputs(
        self,
        prompt: str,
        refer_image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.text2image_inputs(
            prompt=prompt,
            max_seq_length=max_seq_length,
        )
        refer_pixel_values = self._process_reference_image(refer_image)

        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            refer_pixel_values=refer_pixel_values,
        )

    def editing(
        self,
        prompt: str,
        refer_image: Union[Image.Image, str],
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.editing_inputs(
            prompt=prompt,
            refer_image=refer_image,
            max_seq_length=max_seq_length,
        )
        image = self._load_image(image)
        pixel_values = self.vision_processor(image)

        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            refer_pixel_values=prompt_outputs.refer_pixel_values,
            pixel_values=pixel_values,
        )
