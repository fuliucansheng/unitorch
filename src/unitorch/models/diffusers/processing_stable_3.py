# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTokenizer, T5Tokenizer
from torchvision.transforms import (
    Resize,
    CenterCrop,
    RandomCrop,
    Lambda,
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
from torchvision.transforms.functional import crop
from diffusers.image_processor import VaeImageProcessor
from unitorch.models import HfTextClassificationProcessor, GenericOutputs


class Stable3Processor:
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        merge2_path: str,
        vocab3_path: str,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        max_seq_length2: Optional[int] = 256,
        position_start_id: Optional[int] = 0,
        pad_token: Optional[str] = "<|endoftext|>",
        pad_token2: Optional[str] = "!",
        image_size: Optional[int] = 512,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
    ):
        tokenizer1 = CLIPTokenizer(
            vocab_file=vocab_path,
            merges_file=merge_path,
        )

        tokenizer1.cls_token = tokenizer1.bos_token
        tokenizer1.sep_token = tokenizer1.eos_token
        tokenizer1.pad_token = pad_token

        self.text_processor1 = HfTextClassificationProcessor(
            tokenizer=tokenizer1,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

        tokenizer2 = CLIPTokenizer(
            vocab_file=vocab2_path,
            merges_file=merge2_path,
        )

        tokenizer2.cls_token = tokenizer2.bos_token
        tokenizer2.sep_token = tokenizer2.eos_token
        tokenizer2.pad_token = pad_token2

        self.text_processor2 = HfTextClassificationProcessor(
            tokenizer=tokenizer2,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

        tokenizer3 = T5Tokenizer(
            vocab3_path,
        )
        tokenizer3.bos_token_id = 0
        tokenizer3.cls_token = tokenizer3.convert_ids_to_tokens(0)
        tokenizer3.sep_token = tokenizer3.eos_token
        tokenizer3.sep_token_id = tokenizer3.eos_token_id

        self.text_processor3 = HfTextClassificationProcessor(
            tokenizer=tokenizer3,
            max_seq_length=max_seq_length2,
            position_start_id=position_start_id,
        )

        self.image_size = image_size
        self.vision_processor = Compose(
            [
                Resize(image_size),
                CenterCrop(image_size) if center_crop else RandomCrop(image_size),
                RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

        self.condition_vision_processor = Compose(
            [
                Resize(image_size),
                CenterCrop(image_size),
                ToTensor(),
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
            self.vae_mask_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor
            )
            self.vae_condition_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor,
                do_convert_rgb=True,
                do_normalize=False,
            )
        else:
            self.vae_image_processor = None
            self.vae_mask_image_processor = None
            self.vae_condition_image_processor = None

    def text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        prompt3: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_seq_length2: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = self.vision_processor(image)

        prompt2 = prompt2 or prompt
        prompt3 = prompt3 or prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length
        )
        prompt3_outputs = self.text_processor3.classification(
            prompt3, max_seq_length=max_seq_length2
        )

        return GenericOutputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
            input3_ids=prompt3_outputs.input_ids,
            attention3_mask=prompt3_outputs.attention_mask,
        )

    def text2image_inputs(
        self,
        prompt: str,
        prompt2: Optional[str] = None,
        prompt3: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        negative_prompt3: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_seq_length2: Optional[int] = None,
    ):
        prompt2 = prompt2 or prompt
        prompt3 = prompt3 or prompt
        negative_prompt2 = negative_prompt2 or negative_prompt
        negative_prompt3 = negative_prompt3 or negative_prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length
        )
        prompt3_outputs = self.text_processor3.classification(
            prompt3, max_seq_length=max_seq_length2
        )
        negative_prompt_outputs = self.text_processor1.classification(
            negative_prompt, max_seq_length=max_seq_length
        )
        negative_prompt2_outputs = self.text_processor2.classification(
            negative_prompt2, max_seq_length=max_seq_length
        )
        negative_prompt3_outputs = self.text_processor3.classification(
            negative_prompt3, max_seq_length=max_seq_length2
        )
        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
            input3_ids=prompt3_outputs.input_ids,
            attention3_mask=prompt3_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
            negative_input2_ids=negative_prompt2_outputs.input_ids,
            negative_attention2_mask=negative_prompt2_outputs.attention_mask,
            negative_input3_ids=negative_prompt3_outputs.input_ids,
            negative_attention3_mask=negative_prompt3_outputs.attention_mask,
        )

    def image2image_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.vae_image_processor.preprocess(image)[0]

        return GenericOutputs(
            pixel_values=pixel_values,
        )

    def controlnet(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.condition_vision_processor(image)
        return GenericOutputs(pixel_values=pixel_values)

    def controlnet_inputs(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.vae_condition_image_processor.preprocess(image)[0]
        return GenericOutputs(pixel_values=pixel_values)

    def controlnets_inputs(
        self,
        images: List[Union[Image.Image, str]],
    ):
        pixel_values = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            image = image.convert("RGB")

            pixel_values.append(self.vae_condition_image_processor.preprocess(image)[0])

        return GenericOutputs(pixel_values=torch.stack(pixel_values, dim=0))
