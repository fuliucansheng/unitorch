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
from transformers import AutoProcessor

from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE, format_input
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from unitorch.models import GenericOutputs
from unitorch.utils import pop_value


class Flux2Processor:
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
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            subfolder=processor_subfolder,
            token=(use_auth_token if use_auth_token else None),
        )
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
        messages = format_input(
            prompts=[str(prompt)],
            system_message=SYSTEM_MESSAGE,
        )
        outputs = self.processor.apply_chat_template(
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
