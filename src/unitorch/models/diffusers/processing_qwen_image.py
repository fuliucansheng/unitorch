# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
import numpy as np
import torch
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
from transformers import (
    Qwen2VLImageProcessor,
    Qwen2Tokenizer,
    Qwen2TokenizerFast,
    Qwen2_5_VLProcessor,
)
from diffusers.image_processor import VaeImageProcessor

from unitorch.utils import (
    pop_value,
    truncate_sequence_pair,
    read_json_file,
    get_added_token,
)
from unitorch.models import (
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)


class QWenImageProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        vision_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        image_size: Optional[Tuple[int, int]] = None,
        center_crop: Optional[bool] = False,
        random_flip: Optional[bool] = False,
    ):
        """
        Initializes the ClipProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merge file.
            max_seq_length (int, optional): The maximum sequence length for text inputs. Defaults to 262144.
        """
        tokenizer_config = read_json_file(tokenizer_config) if tokenizer_config else {}
        special_tokens_map = (
            read_json_file(special_tokens_map) if special_tokens_map else {}
        )
        added_tokens_decoder = tokenizer_config.pop("added_tokens_decoder", {})
        tokenizer_config = {
            k: (
                get_added_token(v)
                if isinstance(v, dict) and v.get("__type") == "AddedToken"
                else v
            )
            for k, v in tokenizer_config.items()
        }
        tokenizer = Qwen2TokenizerFast(
            vocab_file=vocab_path,
            merges_file=merge_path,
            **tokenizer_config,
        )
        for idx, spec in added_tokens_decoder.items():
            token = spec["content"]
            tokenizer.added_tokens_decoder[idx] = get_added_token(spec)
            tokenizer.added_tokens_encoder[token] = idx

        special_tokens = {}
        for name, spec in special_tokens_map.items():
            if not isinstance(spec, dict or str):
                continue
            special_tokens[name] = get_added_token(spec)
        tokenizer.add_special_tokens(special_tokens)

        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token

        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

        self.image_token = (
            "<|image_pad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

        if vision_config_path is not None:
            self.refer_vision_processor = Qwen2VLImageProcessor.from_json_file(
                vision_config_path
            )
        else:
            self.refer_vision_processor = None

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
            vae_config_dict = json.load(open(vae_config_path))
            vae_scale_factor = 2 ** (
                len(vae_config_dict.get("block_out_channels", [])) - 1
            )
            self.vae_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor * 2
            )
        else:
            self.vae_image_processor = None

        self.prompt_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_editing = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_start_index = 34
        self.prompt_editing_start_index = 64

    def processing_image(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Process images for classification.

        Args:
            images (Image.Image, str, List[Image.Image], List[str]): Input image or list of images.

        Returns:
            GenericOutputs: Processed outputs.
        """
        if isinstance(image, str):
            image = Image.open(image)
        outputs = self.refer_vision_processor(images=[image], return_tensors="pt")
        return outputs

    def text2image_inputs(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_seq_length = max_seq_length + self.prompt_start_index
        prompt = self.prompt_template.format(str(prompt))
        tokens = self.tokenizer.tokenize(prompt)
        tokens = tokens[:max_seq_length]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding = [self.pad_token_id] * (max_seq_length - len(input_ids))
        attention_mask = [1] * len(input_ids) + [0] * len(padding)
        input_ids = input_ids + padding

        if negative_prompt is not None:
            negative_prompt = self.prompt_template.format(str(negative_prompt))
            negative_tokens = self.tokenizer.tokenize(negative_prompt)
            negative_tokens = negative_tokens[:max_seq_length]
            negative_input_ids = self.tokenizer.convert_tokens_to_ids(negative_tokens)
            negative_padding = [self.pad_token_id] * (
                max_seq_length - len(negative_input_ids)
            )
            negative_attention_mask = [1] * len(negative_input_ids) + [0] * len(
                negative_padding
            )
            negative_input_ids = negative_input_ids + negative_padding
        else:
            negative_input_ids = None
            negative_attention_mask = None

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            negative_input_ids=(
                torch.tensor(negative_input_ids, dtype=torch.long)
                if negative_input_ids is not None
                else None
            ),
            negative_attention_mask=(
                torch.tensor(negative_attention_mask, dtype=torch.long)
                if negative_attention_mask is not None
                else None
            ),
        )

    def text2image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.text2image_inputs(
            prompt=prompt,
            negative_prompt=None,
            max_seq_length=max_seq_length,
        )
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
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_seq_length = max_seq_length + self.prompt_editing_start_index
        prompt = self.prompt_template_editing.format(str(prompt))
        image_inputs = self.processing_image(refer_image)
        image_index, image_merge_size = 0, self.refer_vision_processor.merge_size**2
        image_grid_thw = image_inputs["image_grid_thw"]
        while self.image_token in prompt:
            num_image_tokens = image_grid_thw[image_index].prod() // image_merge_size
            prompt = prompt.replace(
                self.image_token,
                "<|placeholder|>" * num_image_tokens,
                1,
            )
            image_index += 1
        prompt = prompt.replace("<|placeholder|>", self.image_token)

        tokens = self.tokenizer.tokenize(prompt)
        tokens = tokens[:max_seq_length]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding = [self.pad_token_id] * (max_seq_length - len(input_ids))
        attention_mask = [1] * len(input_ids) + [0] * len(padding)
        input_ids = input_ids + padding

        if negative_prompt is not None:
            negative_prompt = self.prompt_template_editing.format(str(negative_prompt))
            image_index, image_merge_size = 0, self.refer_vision_processor.merge_size**2
            while self.image_token in negative_prompt:
                num_image_tokens = (
                    image_grid_thw[image_index].prod() // image_merge_size
                )
                negative_prompt = negative_prompt.replace(
                    self.image_token,
                    "<|placeholder|>" * num_image_tokens,
                    1,
                )
                image_index += 1
            negative_prompt = negative_prompt.replace(
                "<|placeholder|>", self.image_token
            )
            negative_tokens = self.tokenizer.tokenize(negative_prompt)
            negative_tokens = negative_tokens[:max_seq_length]
            negative_input_ids = self.tokenizer.convert_tokens_to_ids(negative_tokens)
            negative_padding = [self.pad_token_id] * (
                max_seq_length - len(negative_input_ids)
            )
            negative_attention_mask = [1] * len(negative_input_ids) + [0] * len(
                negative_padding
            )
            negative_input_ids = negative_input_ids + negative_padding
        else:
            negative_input_ids = None
            negative_attention_mask = None

        refer_vae_pixel_values = self.vae_image_processor.preprocess(refer_image)[0]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            negative_input_ids=(
                torch.tensor(negative_input_ids, dtype=torch.long)
                if negative_input_ids is not None
                else None
            ),
            negative_attention_mask=(
                torch.tensor(negative_attention_mask, dtype=torch.long)
                if negative_attention_mask is not None
                else None
            ),
            refer_image_grid_thw=torch.tensor(
                image_inputs["image_grid_thw"], dtype=torch.long
            ),
            refer_pixel_values=torch.tensor(image_inputs["pixel_values"]),
            refer_vae_pixel_values=torch.tensor(refer_vae_pixel_values),
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
            negative_prompt=None,
            max_seq_length=max_seq_length,
        )
        pixel_values = self.vision_processor(image)

        return GenericOutputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            refer_image_grid_thw=prompt_outputs.refer_image_grid_thw,
            refer_pixel_values=prompt_outputs.refer_pixel_values,
            refer_vae_pixel_values=prompt_outputs.refer_vae_pixel_values,
            pixel_values=pixel_values,
        )
