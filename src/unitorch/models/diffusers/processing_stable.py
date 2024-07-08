# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTokenizer, CLIPImageProcessor
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
from diffusers.image_processor import VaeImageProcessor
from unitorch.models import (
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)


class StableProcessor(HfTextClassificationProcessor):
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
        random_flip: Optional[bool] = False,
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
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB")

        pixel_values = self.vision_processor(image)

        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)

        return GenericOutputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
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

    def image2image_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.vae_image_processor.preprocess(image)[0]

        return GenericOutputs(pixel_values=pixel_values)

    def inpainting_inputs(
        self,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        if isinstance(mask_image, str):
            mask_image = Image.open(mask_image)
        mask_image = mask_image.convert("L")

        pixel_values = self.vae_image_processor.preprocess(image)[0]
        pixel_masks = self.vae_image_processor.preprocess(mask_image)[0]
        pixel_masks = (pixel_masks + 1) / 2

        return GenericOutputs(
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
        )

    def resolution_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.vae_image_processor.preprocess(image)[0]

        return GenericOutputs(pixel_values=pixel_values)

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


class StableVideoProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
        vae_config_path: str,
    ):
        vision_processor = CLIPImageProcessor.from_json_file(vision_config_path)
        super().__init__(
            vision_processor=vision_processor,
        )
        vae_config_dict = json.load(open(vae_config_path))
        vae_scale_factor = 2 ** (len(vae_config_dict.get("block_out_channels", [])) - 1)
        self.vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def image2video_inputs(
        self,
        image: Union[Image.Image, str],
        vae_image: Union[Image.Image, str],
    ):
        pixel_outputs = self.classification(
            image=image.resize((224, 224), Image.BICUBIC),
        )
        vae_pixel_values = self.vae_image_processor.preprocess(vae_image)[0]

        return GenericOutputs(
            pixel_values=pixel_outputs.pixel_values,
            vae_pixel_values=vae_pixel_values,
        )
