# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from transformers import BloomTokenizerFast
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import (
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    GenericOutputs,
)


class BloomProcessor(HfTextClassificationProcessor, HfTextGenerationProcessor):
    """
    Processor for the Bloom model that combines text classification and text generation functionality.
    """

    def __init__(
        self,
        tokenizer_file: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initializes a new instance of the BloomProcessor.

        Args:
            tokenizer_file (str): The path to the tokenizer file.
            max_seq_length (Optional[int]): The maximum sequence length for classification. Defaults to 128.
            max_gen_seq_length (Optional[int]): The maximum sequence length for generation. Defaults to 48.
        """
        tokenizer = BloomTokenizerFast(tokenizer_file=tokenizer_file)
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.cls_token_id = tokenizer.bos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
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

    def classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Preprocesses text for classification.

        Args:
            text (str): The input text to classify.
            text_pair (Optional[str]): The second input text for sequence classification. Defaults to None.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The processed input IDs and attention mask tensors.
        """
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

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Preprocesses text as generation inputs.

        Args:
            text (str): The input text for generation.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The processed input IDs tensor.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))[-max_seq_length:]
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
        )

    def _instrution_tokenize(self, instruction, input, max_seq_length):
        tokens1 = self.tokenizer.tokenize(instruction.format(""))
        tokens2 = self.tokenizer.tokenize(str(input))[
            : max_seq_length - len(tokens1) - 2
        ]
        input2 = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens2))
        input = instruction.format(input2)
        return self.tokenizer.tokenize(input)[:max_seq_length]

    def instruction_generation_inputs(
        self,
        instruction: str,
        input: str,
        max_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Preprocesses text as generation inputs.

        Args:
            text (str): The input text for generation.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to None.

        Returns:
            GenericOutputs: The processed input IDs tensor.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self._instrution_tokenize(instruction, input, max_seq_length)
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
        )

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Preprocesses text as generation labels.

        Args:
            text (str): The input text for generation labels.
            max_gen_seq_length (Optional[int]): The maximum generation sequence length. Defaults to None.

        Returns:
            GenericOutputs: The processed input IDs and attention mask tensors.
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
        text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ) -> GenericOutputs:
        """
        Preprocesses text for generation.

        Args:
            text (str): The input text for generation.
            text_pair (str): The second input text for generation.
            max_seq_length (Optional[int]): The maximum sequence length for classification. Defaults to None.
            max_gen_seq_length (Optional[int]): The maximum generation sequence length. Defaults to None.

        Returns:
            GenericOutputs: The processed input IDs and attention mask tensors.
        """
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
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
        )

    def instruction_generation(
        self,
        instruction: str,
        input: str,
        output: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        tokens = self._instrution_tokenize(instruction, input, max_seq_length)
        tokens_pair = self.tokenizer.tokenize(str(output))[: max_gen_seq_length - 1] + [
            self.eos_token
        ]
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
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
        )
