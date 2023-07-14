# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
import logging
from functools import partial
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from transformers import LlamaTokenizer
from transformers import BlipImageProcessor

from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import (
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)


class MiniGPT4Blip2LlamaProcessor(
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
    HfTextGenerationProcessor,
):
    """
    MiniGPT4Blip2LlamaProcessor is a class for processing inputs and outputs of the MiniGPT4 model with Blip2 and Llama.
    It inherits from the _MiniGPT4Blip2LlamaProcessor class.
    """

    def __init__(
        self,
        vocab_file: str,
        vision_config_path: str,
        max_prefix_seq_length: Optional[int] = 64,
        max_suffix_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initializes a MiniGPT4Blip2LlamaProcessor instance.

        Args:
            vocab_path (str): The file path to the vocabulary.
            vision_config_path (str): The file path to the vision configuration.
            max_prefix_seq_length (int, optional): The maximum length of the prefix sequence. Defaults to 64.
            max_suffix_seq_length (int, optional): The maximum length of the suffix sequence. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum length of the generated sequence. Defaults to 48.
        """
        tokenizer = LlamaTokenizer(vocab_file=vocab_file)
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.cls_token_id = tokenizer.bos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.unk_token_id
        vision_processor = BlipImageProcessor.from_json_file(vision_config_path)
        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_prefix_seq_length,
        )
        HfTextGenerationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_prefix_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        HfImageClassificationProcessor.__init__(self, vision_processor=vision_processor)
        self.max_prefix_seq_length = max_prefix_seq_length
        self.max_suffix_seq_length = max_suffix_seq_length

    def prompt(
        self,
        prefix_text: str,
        suffix_text: str,
        image: Image.Image,
        max_prefix_seq_length: Optional[int] = None,
        max_suffix_seq_length: Optional[int] = None,
    ):
        """
        Process text as a prompt.

        Args:
            prefix_text (str): The prefix text.
            suffix_text (str): The suffix text.
            image (PIL.Image.Image): The input image.
            max_prefix_seq_length (int, optional): The maximum length of the prefix sequence. Defaults to None.
            max_suffix_seq_length (int, optional): The maximum length of the suffix sequence. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids tensor.
        """
        max_prefix_seq_length = pop_value(
            max_prefix_seq_length,
            self.max_prefix_seq_length,
        )
        max_suffix_seq_length = pop_value(
            max_suffix_seq_length,
            self.max_suffix_seq_length,
        )
        prefix_tokens = self.tokenizer.tokenize(str(prefix_text))

        if len(prefix_tokens) >= max_prefix_seq_length:
            logging.warning(
                f"Input prefix text {prefix_text} has been truncated to {max_prefix_seq_length - 1} tokens."
            )
            prefix_tokens = prefix_tokens[: max_prefix_seq_length - 1]

        suffix_tokens = self.tokenizer.tokenize(str(suffix_text))

        if len(suffix_tokens) > max_suffix_seq_length:
            logging.warning(
                f"Input suffix text {suffix_text} has been truncated to {max_suffix_seq_length} tokens."
            )
            suffix_tokens = suffix_tokens[:max_suffix_seq_length]

        prefix_tokens = [self.bos_token] + prefix_tokens
        prefix_padding = [self.pad_token] * (max_prefix_seq_length - len(prefix_tokens))
        prefix_tokens = prefix_padding + prefix_tokens
        prefix_input_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)

        suffix_padding = [self.pad_token] * (max_suffix_seq_length - len(suffix_tokens))
        suffix_tokens = suffix_padding + suffix_tokens
        suffix_input_ids = self.tokenizer.convert_tokens_to_ids(suffix_tokens)

        outputs = HfImageClassificationProcessor.classification(
            self,
            image=image,
        )

        return GenericOutputs(
            prefix_input_ids=torch.tensor(prefix_input_ids, dtype=torch.long),
            suffix_input_ids=torch.tensor(suffix_input_ids, dtype=torch.long),
            pixel_values=outputs.pixel_values,
        )

    def generation_inputs(
        self,
        prefix_text: str,
        suffix_text: str,
        image: Image.Image,
        max_prefix_seq_length: Optional[int] = None,
        max_suffix_seq_length: Optional[int] = None,
    ):
        """
        Process text for generation inputs.

        Args:
            prefix_text (str): The prefix text.
            suffix_text (str): The suffix text.
            image (PIL.Image.Image): The input image.
            max_prefix_seq_length (int, optional): The maximum length of the prefix sequence. Defaults to None.
            max_suffix_seq_length (int, optional): The maximum length of the suffix sequence. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids tensor.
        """
        max_prefix_seq_length = pop_value(
            max_prefix_seq_length,
            self.max_prefix_seq_length,
        )
        max_suffix_seq_length = pop_value(
            max_suffix_seq_length,
            self.max_suffix_seq_length,
        )
        prefix_tokens = self.tokenizer.tokenize(str(prefix_text))

        if len(prefix_tokens) >= max_prefix_seq_length:
            logging.warning(
                f"Input prefix text {prefix_text} has been truncated to {max_prefix_seq_length - 1} tokens."
            )
            prefix_tokens = prefix_tokens[: max_prefix_seq_length - 1]

        suffix_tokens = self.tokenizer.tokenize(str(suffix_text))

        if len(suffix_tokens) > max_suffix_seq_length:
            logging.warning(
                f"Input suffix text {suffix_text} has been truncated to {max_suffix_seq_length} tokens."
            )
            suffix_tokens = suffix_tokens[:max_suffix_seq_length]

        prefix_tokens = [self.bos_token] + prefix_tokens
        prefix_padding = [self.pad_token] * (max_prefix_seq_length - len(prefix_tokens))
        prefix_tokens = prefix_padding + prefix_tokens
        prefix_input_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)

        suffix_padding = [self.pad_token] * (max_suffix_seq_length - len(suffix_tokens))
        suffix_tokens = suffix_padding + suffix_tokens
        suffix_input_ids = self.tokenizer.convert_tokens_to_ids(suffix_tokens)

        outputs = HfImageClassificationProcessor.classification(
            self,
            image=image,
        )

        return GenericOutputs(
            prefix_input_ids=torch.tensor(prefix_input_ids, dtype=torch.long),
            suffix_input_ids=torch.tensor(suffix_input_ids, dtype=torch.long),
            pixel_values=outputs.pixel_values,
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
        prefix_text: str,
        suffix_text: str,
        text_pair: str,
        image: Image.Image,
        max_prefix_seq_length: Optional[int] = None,
        max_suffix_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Process text for generation.

        Args:
            prefix_text (str): The prefix text.
            suffix_text (str): The suffix text.
            text_pair (str): The text pair.
            image (PIL.Image.Image): The input image.
            max_prefix_seq_length (int, optional): The maximum length of the prefix sequence. Defaults to None.
            max_suffix_seq_length (int, optional): The maximum length of the suffix sequence. Defaults to None.
            max_gen_seq_length (int, optional): The maximum length of the generated sequence. Defaults to None.

        Returns:
            GenericOutputs: Processed input_ids, attention_mask, input_ids_label, and attention_mask_label tensors.
        """
        max_prefix_seq_length = pop_value(
            max_prefix_seq_length,
            self.max_prefix_seq_length,
        )
        max_suffix_seq_length = pop_value(
            max_suffix_seq_length,
            self.max_suffix_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )
        prefix_tokens = self.tokenizer.tokenize(str(prefix_text))

        if len(prefix_tokens) >= max_prefix_seq_length:
            logging.warning(
                f"Input prefix text {prefix_text} has been truncated to {max_prefix_seq_length - 1} tokens."
            )
            prefix_tokens = prefix_tokens[: max_prefix_seq_length - 1]

        suffix_tokens = self.tokenizer.tokenize(str(suffix_text))

        if len(suffix_tokens) > max_suffix_seq_length:
            logging.warning(
                f"Input suffix text {suffix_text} has been truncated to {max_suffix_seq_length} tokens."
            )
            suffix_tokens = suffix_tokens[:max_suffix_seq_length]

        prefix_tokens = [self.bos_token] + prefix_tokens
        prefix_padding = [self.pad_token] * (max_prefix_seq_length - len(prefix_tokens))
        prefix_attention_mask = [0] * len(prefix_padding) + [1] * len(prefix_tokens)
        prefix_tokens = prefix_padding + prefix_tokens
        prefix_input_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)

        suffix_padding = [self.pad_token] * (max_suffix_seq_length - len(suffix_tokens))
        suffix_attention_mask = [0] * len(suffix_padding) + [1] * len(suffix_tokens)
        suffix_tokens = suffix_padding + suffix_tokens
        suffix_input_ids = self.tokenizer.convert_tokens_to_ids(suffix_tokens)

        tokens_pair = self.tokenizer.tokenize(str(text_pair))[
            : max_gen_seq_length - 1
        ] + [self.eos_token]

        padding_pair = [self.pad_token] * (max_gen_seq_length - len(tokens_pair))
        input_ids_pair = self.tokenizer.convert_tokens_to_ids(
            tokens_pair + padding_pair
        )
        attention_mask_pair = [1] * len(tokens_pair) + [0] * len(padding_pair)

        tokens_label = tokens_pair + [self.pad_token] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        input_ids_label = self.tokenizer.convert_tokens_to_ids(tokens_label)
        input_ids_label = [0] * (max_suffix_seq_length - 1) + input_ids_label
        attention_mask_label = [1] * len(tokens_pair) + [0] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        attention_mask_label = [0] * (max_suffix_seq_length - 1) + attention_mask_label

        outputs = HfImageClassificationProcessor.classification(
            self,
            image=image,
        )

        return GenericOutputs(
            prefix_input_ids=torch.tensor(prefix_input_ids, dtype=torch.long),
            prefix_attention_mask=torch.tensor(prefix_attention_mask, dtype=torch.long),
            suffix_input_ids=torch.tensor(suffix_input_ids, dtype=torch.long),
            suffix_attention_mask=torch.tensor(suffix_attention_mask, dtype=torch.long),
            input_ids_pair=torch.tensor(input_ids_pair, dtype=torch.long),
            attention_mask_pair=torch.tensor(attention_mask_pair, dtype=torch.long),
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
            pixel_values=outputs.pixel_values,
        )
