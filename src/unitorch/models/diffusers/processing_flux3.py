# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.
"""Tensor-only CLI preprocessing for FLUX.3 scene captions and RGB images."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import Qwen2Tokenizer

from unitorch.models import GenericOutputs


class Flux3Processor:
    """Keep source and target separate and preserve image geometry by default.

    Token storage has fixed max_seq_length for generic collation. The model
    trims each batch to its real text window rounded to 80 tokens, preserving
    the official right-padding embeddings. Overlong captions raise an error.
    image_size is (width, height) and validates rather than resizes inputs.
    """

    def __init__(self, tokenizer_path: str, max_seq_length: int = 8192,
                 image_size: Optional[Tuple[int, int]] = None, image_size_multiple: int = 16):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.tokenizer.padding_side = "right"
        template = Path(tokenizer_path) / "chat_template.json"
        if template.exists():
            self.tokenizer.chat_template = json.loads(template.read_text())["chat_template"]
        if not self.tokenizer.chat_template:
            raise ValueError("The official Qwen3-VL chat template is required")
        if max_seq_length < 80 or max_seq_length > 8192:
            raise ValueError("max_seq_length must be between 80 and the official 8192 token limit")
        self.max_seq_length = max_seq_length
        self.image_size = tuple(image_size) if image_size is not None else None
        self.image_size_multiple = image_size_multiple

    def _text(self, caption):
        if not isinstance(caption, str):
            raise TypeError("A scene caption must be a string")
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": caption}],
                                                  tokenize=False, add_generation_prompt=True)
        if len(self.tokenizer(text)["input_ids"]) > self.max_seq_length:
            raise ValueError("Scene caption exceeds max_seq_length tokens; increase the explicit limit")
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                max_length=self.max_seq_length, truncation=False)
        return inputs["input_ids"][0], inputs["attention_mask"][0]

    def _image(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
        else:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if self.image_size and image.size != self.image_size:
            raise ValueError(f"Image size {image.size} does not match requested geometry {self.image_size}; normalize explicitly")
        if any(side <= 0 or side % self.image_size_multiple for side in image.size):
            raise ValueError(f"Image geometry must have sides divisible by {self.image_size_multiple}")
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1

    def text2image_inputs(self, prompt: str, negative_prompt: str = ""):
        ids, mask = self._text(prompt)
        negative_ids, negative_mask = self._text(negative_prompt)
        return GenericOutputs(input_ids=ids, attention_mask=mask,
                              negative_input_ids=negative_ids, negative_attention_mask=negative_mask)

    def text2image(self, prompt, image, negative_prompt=""):
        outputs = self.text2image_inputs(prompt, negative_prompt)
        return GenericOutputs(**outputs, pixel_values=self._image(image))

    def editing_inputs(self, prompt, refer_image, negative_prompt=""):
        outputs = self.text2image_inputs(prompt, negative_prompt)
        if isinstance(refer_image, (list, tuple)):
            if not 1 <= len(refer_image) <= 10:
                raise ValueError("Editing requires 1 to 10 reference images")
            references = torch.stack([self._image(image) for image in refer_image])
        else:
            references = self._image(refer_image)
        return GenericOutputs(**outputs, refer_pixel_values=references)

    def editing(self, prompt, refer_image, image, negative_prompt=""):
        return GenericOutputs(**self.editing_inputs(prompt, refer_image, negative_prompt), pixel_values=self._image(image))
