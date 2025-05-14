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

        self.divisor = 16
        if self.video_size is not None:
            self.video_size = (
                self.video_size[0] // self.divisor * self.divisor,
                self.video_size[1] // self.divisor * self.divisor,
            )
            self.center_crop_processor = CenterCrop(
                size=(self.video_size[1], self.video_size[0])
            )
        else:
            self.center_crop_processor = None
        self.frame_processor = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

        if image_config_path is not None:
            self.vision_processor = CLIPImageProcessor.from_json_file(image_config_path)
        else:
            self.vision_processor = None

    def get_video_frames(self, video: Union[cv2.VideoCapture, str]):
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

        return frames

    def text2video(
        self,
        prompt: str,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        frames = self.get_video_frames(video)

        pixel_values = []
        for frame in frames:
            if self.frame_processor is not None:
                if self.video_size is not None:
                    width, height = frame.size
                    scale = max(self.video_size[0] / width, self.video_size[1] / height)
                    frame = frame.resize(
                        (round(width * scale), round(height * scale)),
                        resample=Image.LANCZOS,
                    )
                    frame = self.center_crop_processor(frame)
                else:
                    width, height = frame.size
                    new_width = width // self.divisor * self.divisor
                    new_height = height // self.divisor * self.divisor
                    frame = frame.resize(
                        (new_width, new_height), resample=Image.LANCZOS
                    )
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
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        image: Optional[Union[Image.Image, str]] = None,
        max_seq_length: Optional[int] = None,
    ):
        frames = self.get_video_frames(video)

        outputs = self.text2video(
            prompt=prompt,
            video=frames,
            max_seq_length=max_seq_length,
        )
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if image is None:
            image = frames[0].convert("RGB")

        if self.video_size is not None:
            width, height = image.size
            scale = max(self.video_size[0] / width, self.video_size[1] / height)
            image = image.resize(
                (round(height * scale), round(width * scale)), resample=Image.LANCZOS
            )
            image = self.center_crop_processor(image)
        else:
            width, height = image.size
            new_width = width // self.divisor * self.divisor
            new_height = height // self.divisor * self.divisor
            image = image.resize((new_width, new_height), resample=Image.LANCZOS)

        condition_pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]

        vae_pixel_values = self.vae_image_processor.preprocess(image)[0]

        return GenericOutputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            condition_pixel_values=condition_pixel_values,
            vae_pixel_values=vae_pixel_values,
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

        if self.video_size is not None:
            width, height = image.size
            scale = max(self.video_size[0] / width, self.video_size[1] / height)
            image = image.resize(
                (round(height * scale), round(width * scale)), resample=Image.LANCZOS
            )
            image = self.center_crop_processor(image)
        else:
            width, height = image.size
            new_width = width // self.divisor * self.divisor
            new_height = height // self.divisor * self.divisor
            image = image.resize((new_width, new_height), resample=Image.LANCZOS)

        vae_pixel_values = self.vae_image_processor.preprocess(image)[0]
        pixel_values = self.vision_processor.preprocess(
            image, return_tensors="pt"
        ).pixel_values[0]
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
