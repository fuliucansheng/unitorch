# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
import numpy as np
import torch
from transformers import (
    Qwen2VLImageProcessor,
    Qwen2Tokenizer,
    Qwen2TokenizerFast,
    Qwen2_5_VLProcessor,
)
from unitorch.utils import (
    pop_value,
    truncate_sequence_pair,
    read_json_file,
    get_added_token,
)
from unitorch.models import (
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfLlmProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)


class QWenVLProcessor(HfLlmProcessor):
    def __init__(
        self,
        tokenizer_file: str,
        vision_config_path: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 1280,
        max_gen_seq_length: Optional[int] = 512,
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
            tokenizer_file=tokenizer_file,
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
        if chat_template:
            tokenizer.chat_template = read_json_file(chat_template)["chat_template"]
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token

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
        self.vision_processor = Qwen2VLImageProcessor.from_json_file(vision_config_path)

        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    def processing_images(
        self,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
    ):
        """
        Process images for classification.

        Args:
            images (Image.Image, str, List[Image.Image], List[str]): Input image or list of images.

        Returns:
            GenericOutputs: Processed outputs.
        """
        if isinstance(images, (Image.Image, str)):
            images = [images]
        images = [
            image if isinstance(image, Image.Image) else Image.open(image)
            for image in images
        ]
        outputs = self.vision_processor(images=images, return_tensors="pt")
        return outputs

    def classification(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        max_seq_length: Optional[int] = None,
    ):
        image_inputs = self.processing_images(images)
        image_index, image_merge_size = 0, self.vision_processor.merge_size**2
        image_grid_thw = image_inputs["image_grid_thw"] if image_inputs else None
        while self.image_token in text:
            num_image_tokens = image_grid_thw[image_index].prod() // image_merge_size
            text = text.replace(
                self.image_token,
                "<|placeholder|>" * num_image_tokens,
                1,
            )
            image_index += 1
        text = str(text).replace("<|placeholder|>", self.image_token)
        text_inputs = super().classification(text, max_seq_length=max_seq_length)

        return GenericOutputs(
            input_ids=torch.tensor(text_inputs.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(text_inputs.attention_mask, dtype=torch.long),
            image_grid_thw=torch.tensor(image_grid_thw, dtype=torch.long),
            pixel_values=torch.tensor(image_inputs["pixel_values"]),
        )

    def generation_inputs(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        max_seq_length: Optional[int] = None,
    ):
        image_inputs = self.processing_images(images) if images else None
        image_index, image_merge_size = 0, self.vision_processor.merge_size**2
        image_grid_thw = image_inputs["image_grid_thw"] if image_inputs else None
        while self.image_token in text:
            num_image_tokens = image_grid_thw[image_index].prod() // image_merge_size
            text = text.replace(
                self.image_token,
                "<|placeholder|>" * num_image_tokens,
                1,
            )
            image_index += 1
        text = str(text).replace("<|placeholder|>", self.image_token)
        text_inputs = super().classification(text, max_seq_length=max_seq_length)
        return GenericOutputs(
            input_ids=torch.tensor(text_inputs.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(text_inputs.attention_mask, dtype=torch.long),
            image_grid_thw=torch.tensor(image_grid_thw, dtype=torch.long),
            pixel_values=torch.tensor(image_inputs["pixel_values"]),
        )

    def generation(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        text, text_pair = str(text), str(text_pair)
        image_inputs = self.processing_images(images) if images else None
        image_index, image_merge_size = 0, self.vision_processor.merge_size**2
        image_grid_thw = image_inputs["image_grid_thw"] if image_inputs else None
        while self.image_token in text:
            num_image_tokens = image_grid_thw[image_index].prod() // image_merge_size
            text = text.replace(
                self.image_token,
                "<|placeholder|>" * num_image_tokens,
                1,
            )
            image_index += 1
        text = text.replace("<|placeholder|>", self.image_token)

        text_inputs = super().generation(
            text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

        return GenericOutputs(
            input_ids=torch.tensor(text_inputs.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(text_inputs.attention_mask, dtype=torch.long),
            image_grid_thw=(
                torch.tensor(image_grid_thw, dtype=torch.long)
                if image_grid_thw is not None
                else None
            ),
            pixel_values=(
                torch.tensor(image_inputs["pixel_values"])
                if image_inputs is not None
                else None
            ),
            input_ids_label=torch.tensor(text_inputs.input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(
                text_inputs.attention_mask_label, dtype=torch.long
            ),
        )

    def messages_generation(
        self,
        messages: List[Dict[str, Any]],
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Preprocesses messages for generation.

        Args:
            messages (List[Dict[str, Any]]): The list of messages to process.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The processed input IDs tensor.
        """
        while messages and messages[-1]["role"] != "assistant":
            messages.pop()

        text = self.chat_template(messages[:-1])
        text_pair = self.chat_template(messages[-1:])
        outputs = self.generation(
            text=text,
            images=images,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return GenericOutputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            image_grid_thw=outputs.image_grid_thw,
            pixel_values=outputs.pixel_values,
            input_ids_label=outputs.input_ids_label,
            attention_mask_label=outputs.attention_mask_label,
        )
