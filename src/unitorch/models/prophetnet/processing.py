# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from functools import partial

import torch
from transformers import ProphetNetTokenizer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import HfTextGenerationProcessor


def get_prophetnet_tokenizer(vocab_path, special_input_ids=dict()):
    """
    Get the tokenizer for ProphetNet model.

    Args:
        vocab_path (str): The path to the vocabulary file.
        special_input_ids (Dict, optional): Special input IDs mapping. Defaults to an empty dictionary.

    Returns:
        ProphetNetTokenizer: The tokenizer object.
    """
    assert os.path.exists(vocab_path)
    tokenizer = ProphetNetTokenizer(vocab_path)
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class ProphetNetProcessor(HfTextGenerationProcessor):
    """
    Processor for ProphetNet-based text generation models.
    """

    def __init__(
        self,
        vocab_path: str,
        special_input_ids: Optional[Dict] = dict(),
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initializes the ProphetNetProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            special_input_ids (Optional[Dict]): Special input IDs. Defaults to an empty dictionary.
            max_seq_length (Optional[int]): Maximum sequence length. Defaults to 128.
            max_gen_seq_length (Optional[int]): Maximum generated sequence length. Defaults to 48.
        """
        tokenizer = get_prophetnet_tokenizer(
            vocab_path,
            special_input_ids=special_input_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
