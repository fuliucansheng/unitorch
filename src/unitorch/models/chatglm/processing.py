# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from unitorch.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import (
    HfTextClassificationProcessor,
    HfTextGenerationProcessor,
    GenericOutputs,
)


class ChatGLMProcessor(HfTextClassificationProcessor, HfTextGenerationProcessor):
    """
    Processor for handling text classification and text generation tasks in a chat-based conversational model.

    Args:
        vocab_file (str): Path to the vocabulary file.
        tokenizer_file (str): Path to the tokenizer file.
        max_seq_length (int, optional): Maximum sequence length for text classification. Defaults to 128.
        max_gen_seq_length (int, optional): Maximum sequence length for text generation. Defaults to 48.
    """

    def __init__(
        self,
        vocab_file: str,
        tokenizer_file: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        if not os.path.exists(tokenizer_file):
            config_dict = {}
        else:
            config_dict = json.loads(open(tokenizer_file, "r").read())
        tokenizer = ChatGLMTokenizer(vocab_file=vocab_file, **config_dict)
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
        self.gmask_token = self.tokenizer.gmask_token

    def classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Preprocesses text for classification task.

        Args:
            text (str): The input text to classify.
            text_pair (str, optional): The second input text for sequence pair classification. Defaults to None.
            max_seq_length (int, optional): Maximum sequence length. Overrides the default value set during initialization.

        Returns:
            GenericOutputs: Preprocessed input for classification task.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 2]
            tokens = tokens + [self.gmask_token] + [self.bos_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            truncate_sequence_pair(tokens, tokens_pair, max_seq_length - 2)
            tokens = tokens + tokens_pair + [self.gmask_token, self.bos_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids = len(padding) * [self.pad_token_id] + input_ids

        attention_mask = torch.ones(max_seq_length, max_seq_length, dtype=torch.long)
        attention_mask[len(padding) :, len(padding) : max_seq_length - 1] = 0
        attention_mask[max_seq_length - 1, max_seq_length - 1] = 0

        position_ids1 = [0] * len(padding) + list(range(0, len(tokens)))
        position_ids2 = [0] * (max_seq_length - 1) + [1]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.bool).unsqueeze(0),
            position_ids=torch.tensor(
                [position_ids1, position_ids2],
                dtype=torch.long,
            ),
        )

    def prompt(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Preprocesses input text for prompt.

        Args:
            text (str): The input text for the prompt.
            max_seq_length (int, optional): Maximum sequence length for classification. Overrides the default value set during initialization.

        Returns:
            GenericOutputs: Preprocessed input for prompt.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))[: max_seq_length - 2] + [
            self.gmask_token,
            self.bos_token,
        ]
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
        )

    def generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Preprocesses input text for generation prompt.

        Args:
            text (str): The input text for the generation prompt.
            max_seq_length (int, optional): Maximum sequence length for generation. Overrides the default value set during initialization.

        Returns:
            GenericOutputs: Preprocessed input for generation prompt.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self.tokenizer.tokenize(str(text))[: max_seq_length - 2] + [
            self.gmask_token,
            self.bos_token,
        ]
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
        )

    def generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Preprocesses input text for generation labels.

        Args:
            text (str): The input text for generation labels.
            max_gen_seq_length (int, optional): Maximum sequence length for generation labels. Overrides the default value set during initialization.

        Returns:
            GenericOutputs: Preprocessed input for generation labels.
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
    ):
        """
        Preprocesses text for generation task.

        Args:
            text (str): The input text for the generation prompt.
            text_pair (str): The input text for conditioning or continuation.
            max_seq_length (int, optional): Maximum sequence length for classification. Overrides the default value set during initialization.
            max_gen_seq_length (int, optional): Maximum sequence length for generation. Overrides the default value set during initialization.

        Returns:
            GenericOutputs: Preprocessed input for generation task.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        tokens = self.tokenizer.tokenize(str(text))[1 - max_seq_length :] + [
            self.gmask_token
        ]
        tokens_pair = (
            [self.bos_token]
            + self.tokenizer.tokenize(str(text_pair))[: max_gen_seq_length - 2]
            + [self.eos_token]
        )
        padding_a = [self.pad_token] * (max_seq_length - len(tokens))
        padding_b = [self.pad_token] * (max_gen_seq_length - len(tokens_pair))
        _tokens = padding_a + tokens + tokens_pair + padding_b
        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)

        tokens_label = tokens_pair[1:max_gen_seq_length] + [self.pad_token] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        input_ids_label = self.tokenizer.convert_tokens_to_ids(tokens_label)
        input_ids_label = [0] * max_seq_length + input_ids_label
        attention_mask_label = [1] * len(tokens_pair[1:max_gen_seq_length]) + [0] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        attention_mask_label = [0] * max_seq_length + attention_mask_label

        attention_mask = torch.ones(
            max_seq_length + max_gen_seq_length,
            max_seq_length + max_gen_seq_length,
            dtype=torch.long,
        )
        attention_mask[
            len(padding_a) : max_seq_length + len(tokens_pair),
            len(padding_a) : len(padding_a) + len(tokens),
        ] = 0
        attention_mask[
            max_seq_length : max_seq_length + len(tokens_pair),
            max_seq_length : max_seq_length + len(tokens_pair),
        ] = 1 - torch.tril(torch.ones(len(tokens_pair), len(tokens_pair)))
        position_ids_a = (
            [0] * len(padding_a)
            + list(range(0, len(tokens)))
            + [max_seq_length] * max_gen_seq_length
        )
        position_ids_b = (
            [0] * max_seq_length
            + list(range(1, len(tokens_pair)))
            + [len(tokens_pair)] * (max_gen_seq_length - len(tokens_pair) + 1)
        )

        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.bool).unsqueeze(0),
            position_ids=torch.tensor(
                [position_ids_a, position_ids_b],
                dtype=torch.long,
            ),
            input_ids_label=torch.tensor(input_ids_label, dtype=torch.long),
            attention_mask_label=torch.tensor(attention_mask_label, dtype=torch.long),
        )
