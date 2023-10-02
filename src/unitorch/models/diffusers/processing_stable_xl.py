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
from torchvision.transforms.functional import crop
from diffusers.image_processor import VaeImageProcessor
from unitorch.models import HfTextClassificationProcessor, GenericOutputs


class StableXLProcessor:
    def __init__(
        self,
        vocab1_path: str,
        merge1_path: str,
        vocab2_path: str,
        merge2_path: str,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        image_size: Optional[int] = 512,
        center_crop: Optional[bool] = False,
    ):
        tokenizer1 = CLIPTokenizer(
            vocab_file=vocab1_path,
            merges_file=merge1_path,
        )

        tokenizer1.cls_token = tokenizer1.bos_token
        tokenizer1.sep_token = tokenizer1.eos_token

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

        self.image_size = image_size
        self.center_crop = center_crop
        self.vision_resize = Resize(image_size)
        self.vision_crop = (
            CenterCrop(image_size) if center_crop else RandomCrop(image_size)
        )
        self.vision_flip = RandomHorizontalFlip(p=1.0)
        self.vision_processor = Compose(
            [
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

        self.text_processor2 = HfTextClassificationProcessor(
            tokenizer=tokenizer2,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

        if vae_config_path is not None:
            vae_config_dict = json.load(open(vae_config_path))
            vae_scale_factor = 2 ** (
                len(vae_config_dict.get("block_out_channels", [])) - 1
            )
            self.vae_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor
            )

    def text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        prompt2 = prompt2 or prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length
        )
        original_size = image.size
        image = self.vision_resize(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.image_size) / 2.0)))
            x1 = max(0, int(round((image.width - self.image_size) / 2.0)))
            image = self.vision_crop(image)
        else:
            y1, x1, h, w = self.vision_crop.get_params(
                image, (self.image_size, self.image_size)
            )
            image = crop(image, y1, x1, h, w)
        if self.vision_flip:
            x1 = image.width - x1
            image = self.vision_flip(image)
        crop_top_left = (y1, x1)
        pixel_values = self.vision_processor(image)

        add_time_ids = (
            original_size + crop_top_left + [self.image_size, self.image_size]
        )

        return GenericOutputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
            add_time_ids=torch.tensor(add_time_ids),
        )

    def text2image_inputs(
        self,
        prompt: str,
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        prompt2 = prompt2 or prompt
        negative_prompt2 = negative_prompt2 or negative_prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length
        )
        negative_prompt_outputs = self.text_processor1.classification(
            negative_prompt, max_seq_length=max_seq_length
        )
        negative_prompt2_outputs = self.text_processor2.classification(
            negative_prompt2, max_seq_length=max_seq_length
        )
        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
            negative_input2_ids=negative_prompt2_outputs.input_ids,
            negative_attention2_mask=negative_prompt2_outputs.attention_mask,
        )

    def image2image_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        prompt2 = prompt2 or prompt
        negative_prompt2 = negative_prompt2 or negative_prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length
        )
        negative_prompt_outputs = self.text_processor1.classification(
            negative_prompt, max_seq_length=max_seq_length
        )
        negative_prompt2_outputs = self.text_processor2.classification(
            negative_prompt2, max_seq_length=max_seq_length
        )
        pixel_values = self.vae_image_processor.preprocess(image)[0]
        return GenericOutputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
            negative_input2_ids=negative_prompt2_outputs.input_ids,
            negative_attention2_mask=negative_prompt2_outputs.attention_mask,
        )

    def inpainting_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(mask_image, str):
            mask_image = Image.open(mask_image).convert("L")

        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length
        )
        negative_prompt_outputs = self.text_processor1.classification(
            negative_prompt, max_seq_length=max_seq_length
        )
        negative_prompt2_outputs = self.text_processor2.classification(
            negative_prompt2, max_seq_length=max_seq_length
        )
        pixel_values = self.vae_image_processor.preprocess(image)[0]
        pixel_masks = self.vae_image_processor.preprocess(mask_image)[0]
        pixel_masks = (pixel_masks + 1) / 2
        return GenericOutputs(
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
            negative_input2_ids=negative_prompt2_outputs.input_ids,
            negative_attention2_mask=negative_prompt2_outputs.attention_mask,
        )
