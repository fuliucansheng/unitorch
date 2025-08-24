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
    HfImageClassificationProcessor,
    GenericOutputs,
)


class QWenVLProcessor(
    HfTextGenerationProcessor,
    HfTextClassificationProcessor,
):
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

        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

        HfTextGenerationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    def chat_template(
        self,
        messages: List[Dict[str, Any]],
    ):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text

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
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        text = str(text)
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
        text = text.replace("<|placeholder|>", self.image_token)
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:max_seq_length]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding = [0] * (max_seq_length - len(input_ids))
        attention_mask = [0] * len(padding) + [1] * len(input_ids)
        input_ids = len(padding) * [self.pad_token_id] + input_ids

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            image_grid_thw=torch.tensor(image_grid_thw, dtype=torch.long),
            pixel_values=torch.tensor(image_inputs["pixel_values"]),
        )

    def generation_inputs(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        max_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        text = str(text)
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
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[-max_seq_length:]
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        attention_mask = [0] * len(padding) + [1] * len(tokens)
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            image_grid_thw=torch.tensor(image_grid_thw, dtype=torch.long),
            pixel_values=torch.tensor(image_inputs["pixel_values"]),
        )

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Process text for generation labels.

        Args:
            text (str): Input text.
            max_gen_seq_length (int, optional): Maximum generation sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids and attention_mask tensors.
        """
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))[: max_gen_seq_length - 1] + [
            self.eos_token
        ]
        padding = [self.pad_token] * (max_gen_seq_length - len(tokens))
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_gen_seq_length - len(input_ids))
        input_ids += [self.pad_token_id] * len(padding)
        attention_mask += padding

        assert len(input_ids) == max_gen_seq_length
        assert len(attention_mask) == max_gen_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
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

        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        tokens = self.tokenizer.tokenize(str(text))[-max_seq_length:]
        tokens_pair = self.tokenizer.tokenize(str(text_pair))[
            : max_gen_seq_length - 1
        ] + [self.eos_token]
        padding_a = [self.pad_token] * (max_seq_length - len(tokens))
        padding_b = [self.pad_token] * (max_gen_seq_length - len(tokens_pair))
        attention_mask = (
            [0] * len(padding_a)
            + [1] * (len(tokens) + len(tokens_pair))
            + [0] * len(padding_b)
        )
        _tokens = padding_a + tokens + tokens_pair + padding_b
        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)

        tokens_label = tokens_pair + [self.pad_token] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        input_ids_label = self.tokenizer.convert_tokens_to_ids(tokens_label)
        input_ids_label = [0] * (max_seq_length - 1) + input_ids_label
        attention_mask_label = [1] * len(tokens_pair) + [0] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        attention_mask_label = [0] * (max_seq_length - 1) + attention_mask_label

        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
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
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
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
