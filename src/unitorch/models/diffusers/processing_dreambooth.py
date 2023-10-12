# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTokenizer
from torchvision.transforms import (
    Resize,
    CenterCrop,
    RandomCrop,
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
from diffusers.image_processor import VaeImageProcessor
from unitorch.models import HfTextClassificationProcessor, GenericOutputs


class DreamboothProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        pad_token: Optional[str] = "<|endoftext|>",
        image_size: Optional[int] = 512,
        center_crop: Optional[bool] = False,
    ):
        tokenizer = CLIPTokenizer(
            vocab_file=vocab_path,
            merges_file=merge_path,
        )

        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.pad_token = pad_token

        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

        self.vision_processor = Compose(
            [
                Resize(image_size),
                CenterCrop(image_size) if center_crop else RandomCrop(image_size),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

        if vae_config_path is not None:
            vae_config_dict = json.load(open(vae_config_path))
            vae_scale_factor = 2 ** (
                len(vae_config_dict.get("block_out_channels", [])) - 1
            )
            self.vae_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor
            )
        else:
            self.vae_image_processor = None

    def text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        class_prompt: Optional[str] = None,
        class_image: Optional[Union[Image.Image, str]] = None,
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)
        pixel_values = self.vision_processor(image)

        if class_prompt is not None:
            class_prompt_outputs = self.classification(
                class_prompt, max_seq_length=max_seq_length
            )
            if isinstance(class_image, str):
                class_image = Image.open(class_image)
            class_image = class_image.convert("RGB")
            class_pixel_values = self.vision_processor(class_image)
            return GenericOutputs(
                input_ids=prompt_outputs.input_ids,
                attention_mask=prompt_outputs.attention_mask,
                pixel_values=pixel_values,
                class_input_ids=class_prompt_outputs.input_ids,
                class_attention_mask=class_prompt_outputs.attention_mask,
                class_pixel_values=class_pixel_values,
            )

        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            pixel_values=pixel_values,
        )

    def text2image_inputs(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)
        negative_prompt_outputs = self.classification(
            negative_prompt, max_seq_length=max_seq_length
        )
        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
        )
