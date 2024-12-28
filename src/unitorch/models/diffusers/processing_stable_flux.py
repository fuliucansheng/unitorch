# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTokenizer, T5Tokenizer, SiglipImageProcessor
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


class StableFluxProcessor:
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        vae_config_path: Optional[str] = None,
        redux_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        max_seq_length2: Optional[int] = 256,
        position_start_id: Optional[int] = 0,
        pad_token: Optional[str] = "<|endoftext|>",
        image_size: Optional[Tuple[int, int]] = None,
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

        tokenizer2 = T5Tokenizer(
            vocab2_path,
        )
        tokenizer2.bos_token_id = 0
        tokenizer2.cls_token = tokenizer2.convert_ids_to_tokens(0)
        tokenizer2.sep_token = tokenizer2.eos_token
        tokenizer2.sep_token_id = tokenizer2.eos_token_id

        self.text_processor2 = HfTextClassificationProcessor(
            tokenizer=tokenizer2,
            max_seq_length=max_seq_length2,
            position_start_id=position_start_id,
        )

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
                    CenterCrop((self.image_size[1], self.image_size[0]))
                    if center_crop
                    else RandomCrop((self.image_size[1], self.image_size[0])),
                    RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
                    ToTensor(),
                    Normalize([0.5], [0.5]),
                ]
            )

            self.condition_vision_processor = Compose(
                [
                    Resize((self.image_size[1], self.image_size[0])),
                    CenterCrop((self.image_size[1], self.image_size[0])),
                    ToTensor(),
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

            self.condition_vision_processor = Compose(
                [
                    ToTensor(),
                ]
            )

        if vae_config_path is not None:
            vae_config_dict = json.load(open(vae_config_path))
            vae_scale_factor = 2 ** (
                len(vae_config_dict.get("block_out_channels", [])) - 1
            )
            self.vae_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor * 2
            )
        else:
            self.vae_image_processor = None

        if redux_config_path is not None:
            self.redux_image_processor = SiglipImageProcessor.from_json_file(
                redux_config_path
            )
        else:
            self.redux_image_processor = None

        self.divisor = 16

    def text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_seq_length2: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = self.vision_processor(image)

        prompt2 = prompt2 or prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length2
        )

        return GenericOutputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            input2_ids=prompt2_outputs.input_ids,
            attention2_mask=prompt2_outputs.attention_mask,
        )

    def text2image_inputs(
        self,
        prompt: str,
        prompt2: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        negative_prompt2: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_seq_length2: Optional[int] = None,
    ):
        prompt2 = prompt2 or prompt
        negative_prompt2 = negative_prompt2 or negative_prompt
        prompt_outputs = self.text_processor1.classification(
            prompt, max_seq_length=max_seq_length
        )
        prompt2_outputs = self.text_processor2.classification(
            prompt2, max_seq_length=max_seq_length2
        )
        negative_prompt_outputs = self.text_processor1.classification(
            negative_prompt, max_seq_length=max_seq_length
        )
        negative_prompt2_outputs = self.text_processor2.classification(
            negative_prompt2, max_seq_length=max_seq_length2
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
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        size = image.size if self.image_size is None else self.image_size
        size = (
            size[0] // self.divisor * self.divisor,
            size[1] // self.divisor * self.divisor,
        )
        image = image.resize(size)

        pixel_values = self.vae_image_processor.preprocess(image)[0]

        return GenericOutputs(
            pixel_values=pixel_values,
        )

    def redux_image_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = self.redux_image_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]

        return GenericOutputs(
            pixel_values=pixel_values,
        )

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

        size = image.size if self.image_size is None else self.image_size
        size = (
            size[0] // self.divisor * self.divisor,
            size[1] // self.divisor * self.divisor,
        )
        image = image.resize(size)
        mask_image = mask_image.resize(size)

        pixel_values = self.vae_image_processor.preprocess(image)[0]
        pixel_masks = self.vae_image_processor.preprocess(mask_image)[0]
        pixel_masks = (pixel_masks + 1) / 2

        return GenericOutputs(
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
        )

    def controlnet_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        size = image.size if self.image_size is None else self.image_size
        size = (
            size[0] // self.divisor * self.divisor,
            size[1] // self.divisor * self.divisor,
        )
        image = image.resize(size)

        pixel_values = self.vae_image_processor.preprocess(image)[0]
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
            size = image.size if self.image_size is None else self.image_size
            size = (
                size[0] // self.divisor * self.divisor,
                size[1] // self.divisor * self.divisor,
            )
            image = image.resize(size)

            pixel_values.append(self.vae_image_processor.preprocess(image)[0])

        return GenericOutputs(pixel_values=torch.stack(pixel_values, dim=0))

    def inpainting_control_inputs(
        self,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        size = image.size if self.image_size is None else self.image_size
        size = (
            size[0] // self.divisor * self.divisor,
            size[1] // self.divisor * self.divisor,
        )
        image = image.resize(size)

        if isinstance(mask_image, str):
            mask_image = Image.open(mask_image)
        mask_image = mask_image.convert("L")
        mask_image = mask_image.resize(size)

        pixel_values = self.vae_image_processor.preprocess(image)[0]
        pixel_masks = self.vae_image_processor.preprocess(mask_image)[0]
        pixel_values[:, pixel_masks[0] > 0.5] = -1.0
        return GenericOutputs(pixel_values=pixel_values)

    def adapter_inputs(
        self,
        image: Union[Image.Image, str],
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        size = image.size if self.image_size is None else self.image_size
        size = (
            size[0] // self.divisor * self.divisor,
            size[1] // self.divisor * self.divisor,
        )
        image = image.resize(size)

        pixel_values = self.vae_image_processor.preprocess(image)[0]

        return GenericOutputs(pixel_values=pixel_values)

    def adapters_inputs(
        self,
        images: List[Union[Image.Image, str]],
    ):
        pixel_values = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            image = image.convert("RGB")
            size = image.size if self.image_size is None else self.image_size
            size = (
                size[0] // self.divisor * self.divisor,
                size[1] // self.divisor * self.divisor,
            )
            image = image.resize(size)

            pixel_values.append(self.vae_image_processor.preprocess(image)[0])

        return GenericOutputs(pixel_values=torch.stack(pixel_values, dim=0))
