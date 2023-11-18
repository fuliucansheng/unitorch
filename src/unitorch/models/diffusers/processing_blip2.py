# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import BertTokenizer, CLIPTokenizer
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
from diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from unitorch.models import (
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    GenericOutputs,
)


class Blip2Processor(HfImageClassificationProcessor, HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
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

        vision_processor = BlipImageProcessor.from_json_file(vision_config_path)

        HfImageClassificationProcessor.__init__(
            self,
            vision_processor=vision_processor,
        )

        self.refer_text_processor = HfTextClassificationProcessor(
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
            max_seq_length=max_seq_length,
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

        self.vae_condition_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def text2image(
        self,
    ):
        raise NotImplementedError

    def text2image_inputs(
        self,
        prompt: str,
        refer_prompt: str,
        refer_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
        max_qtokens: Optional[int] = 16,
    ):
        if isinstance(refer_image, str):
            refer_image = Image.open(refer_image).convert("RGB")

        max_seq_length = max_seq_length or self.max_seq_length
        prompt_outputs = HfTextClassificationProcessor.classification(
            self,
            text=prompt,
            max_seq_length=max_seq_length - max_qtokens,
        )
        refer_prompt_outputs = self.refer_text_processor.classification(
            text=refer_prompt,
            max_seq_length=max_seq_length,
        )
        refer_image_outputs = HfImageClassificationProcessor.classification(
            self,
            image=refer_image,
        )
        negative_prompt_outputs = HfTextClassificationProcessor.classification(
            self,
            text=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
            refer_input_ids=refer_prompt_outputs.input_ids,
            refer_attention_mask=refer_prompt_outputs.attention_mask,
            refer_pixel_values=refer_image_outputs.pixel_values,
        )

    def text2image_controlnet_inputs(
        self,
        prompt: str,
        condition_image: Union[Image.Image, str],
        refer_prompt: str,
        refer_image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
        max_qtokens: Optional[int] = 16,
    ):
        if isinstance(condition_image, str):
            condition_image = Image.open(condition_image)

        text2image_inputs = self.text2image_inputs(
            prompt=prompt,
            refer_prompt=refer_prompt,
            refer_image=refer_image,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
            max_qtokens=max_qtokens,
        )

        condition_pixel_values = self.vae_condition_image_processor.preprocess(
            condition_image
        )[0]

        return GenericOutputs(
            input_ids=text2image_inputs.input_ids,
            attention_mask=text2image_inputs.attention_mask,
            negative_input_ids=text2image_inputs.negative_input_ids,
            negative_attention_mask=text2image_inputs.negative_attention_mask,
            refer_input_ids=text2image_inputs.refer_input_ids,
            refer_attention_mask=text2image_inputs.refer_attention_mask,
            refer_pixel_values=text2image_inputs.refer_pixel_values,
            condition_pixel_values=condition_pixel_values,
        )
