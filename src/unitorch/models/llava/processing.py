# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from PIL import Image
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from transformers import LlamaTokenizer, CLIPImageProcessor
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import (
    HfImageClassificationProcessor,
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    GenericOutputs,
)


class LlavaMistralClipProcessor(
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfImageClassificationProcessor,
):
    def __init__(
        self,
        vocab_path: str,
        vision_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initialize the LlamaProcessor.

        Args:
            vocab_file (str): Path to the vocabulary file.
            max_seq_length (int, optional): Maximum sequence length for text classification. Defaults to 128.
            max_gen_seq_length (int, optional): Maximum sequence length for text generation. Defaults to 48.
        """
        tokenizer = LlamaTokenizer(vocab_file=vocab_path)
        tokenizer.add_tokens("<image>", special_tokens=True)
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.cls_token_id = tokenizer.bos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.unk_token_id
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

        if vision_config_path is not None:
            vision_processor = CLIPImageProcessor.from_json_file(vision_config_path)
        else:
            vision_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        HfImageClassificationProcessor.__init__(
            self,
            vision_processor=vision_processor,
        )

    def image_classification(
        self,
        image: Union[Image.Image, str],
    ):
        """
        Performs image classification.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            GenericOutputs: An object containing the processed inputs.
        """
        outputs = HfImageClassificationProcessor.classification(
            self,
            image=image,
        )

        return GenericOutputs(
            pixel_values=outputs.pixel_values,
        )

    def classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Process text for classification.

        Args:
            text (str): Input text.
            text_pair (str, optional): Input text pair. Defaults to None.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids and attention_mask tensors.
        """
        assert "<image>" in text and not text.endswith("<image>")
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[:max_seq_length]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            truncate_sequence_pair(tokens, tokens_pair, max_seq_length)
            tokens = tokens + tokens_pair
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding = [0] * (max_seq_length - len(input_ids))
        attention_mask = [0] * len(padding) + [1] * len(input_ids)
        input_ids = len(padding) * [self.pad_token_id] + input_ids

        pixel_outputs = self.image_classification(
            image=image,
        )

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            pixel_values=pixel_outputs.pixel_values,
        )

    def generation_inputs(
        self,
        text: str,
        image: Union[Image.Image, str],
        max_seq_length: Optional[int] = None,
    ):
        """
        Process text for generation inputs.

        Args:
            text (str): Input text.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids tensor.
        """
        assert "<image>" in text and not text.endswith("<image>")
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = [self.bos_token] + self.tokenizer.tokenize(str(text))[
            1 - max_seq_length :
        ]
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        attention_mask = [0] * len(padding) + [1] * len(tokens)
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        pixel_outputs = self.image_classification(
            image=image,
        )

        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            pixel_values=pixel_outputs.pixel_values,
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
        image: Union[Image.Image, str],
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Process text for generation.

        Args:
            text (str): Input text.
            text_pair (str): Input text pair.
            max_seq_length (int, optional): Maximum sequence length. Defaults to None.
            max_gen_seq_length (int, optional): Maximum generation sequence length. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids, attention_mask, input_ids_label, and attention_mask_label tensors.
        """
        assert "<image>" in text and not text.endswith("<image>")
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        tokens = [self.bos_token] + self.tokenizer.tokenize(str(text))[
            1 - max_seq_length :
        ]
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

        pixel_outputs = self.image_classification(
            image=image,
        )

        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            pixel_values=pixel_outputs.pixel_values,
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
        )
