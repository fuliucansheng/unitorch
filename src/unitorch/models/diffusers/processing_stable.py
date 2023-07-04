# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer
from torchvision.transforms import (
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import HfTextClassificationProcessor, GenericOutputs


class StableProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
        pixel_mean: List[float] = [0.5],
        pixel_std: List[float] = [0.5],
        resize_shape: Optional[List[int]] = [224, 224],
        crop_shape: Optional[List[int]] = [224, 224],
    ):
        if vocab_path is not None and merge_path is not None:
            tokenizer = CLIPTokenizer(
                vocab_file=vocab_path,
                merges_file=merge_path,
            )

            tokenizer.cls_token = tokenizer.bos_token
            tokenizer.sep_token = tokenizer.eos_token

            HfTextClassificationProcessor.__init__(
                self,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                position_start_id=position_start_id,
            )

        self.pixel_mean = torch.tensor(pixel_mean)
        self.pixel_std = torch.tensor(pixel_std)
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.image_processor = Compose(
            [
                Resize(self.resize_shape),
                CenterCrop(self.crop_shape),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(self.pixel_mean, self.pixel_std),
            ]
        )

    def diffusion(
        self,
        prompt: str,
        image: Image.Image,
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)
        pixel_values = self.image_processor(image)
        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            pixel_values=pixel_values,
        )

    def diffusion_inputs(
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
