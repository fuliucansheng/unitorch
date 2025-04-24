# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTokenizer, T5Tokenizer, CLIPImageProcessor
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


class WanProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        vae_config_path: Optional[str] = None,
        image_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        video_size: Optional[Tuple[int, int]] = None,
    ):
        tokenizer = T5Tokenizer(
            vocab_file=vocab_path,
        )

        tokenizer.bos_token_id = 0
        tokenizer.cls_token = tokenizer.convert_ids_to_tokens(0)
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.sep_token_id = tokenizer.eos_token_id

        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

        if video_size is not None:
            self.video_size = (
                video_size
                if isinstance(video_size, tuple)
                else (video_size, video_size)
            )
        else:
            self.video_size = None

        if self.video_size is not None:
            self.frame_processor = Compose(
                [
                    CenterCrop(size=self.video_size),
                    Resize(size=self.video_size, antialias=True),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.frame_processor = None

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

        if image_config_path is not None:
            self.vision_processor = CLIPImageProcessor.from_json_file(image_config_path)
        else:
            self.vision_processor = None
        self.divisor = 8

    def text2video(
        self,
        prompt: str,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(video, str):
            video = cv2.VideoCapture(video)

        if isinstance(video, cv2.VideoCapture):
            frames = []
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                frames.append(pil_img)
        else:
            frames = video

        pixel_values = []
        for frame in frames:
            if self.frame_processor is not None:
                pixel_frame = self.frame_processor(frame)
                pixel_values.append(pixel_frame)
            else:
                raise ValueError(
                    "frame_processor is None, please set video_size to process video"
                )

        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_values = pixel_values.permute(1, 0, 2, 3)

        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)

        return GenericOutputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
        )

    def image2video(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = self.text2video(
            prompt=prompt,
            video=video,
            max_seq_length=max_seq_length,
        )
        condition_pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]

        return GenericOutputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            condition_pixel_values=condition_pixel_values,
        )

    def text2video_inputs(
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

    def image2video_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]

        size = image.size if self.video_size is None else self.video_size
        size = (
            size[0] // self.divisor * self.divisor,
            size[1] // self.divisor * self.divisor,
        )
        image = image.resize(size, resample=Image.LANCZOS)

        vae_pixel_values = self.vae_image_processor.preprocess(image)[0]
        text_outputs = self.text2video_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )

        return GenericOutputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
            condition_pixel_values=pixel_values,
            vae_pixel_values=vae_pixel_values,
        )
