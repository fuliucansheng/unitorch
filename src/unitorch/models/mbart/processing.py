# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import MBartTokenizer

from unitorch.utils import pop_value
from unitorch.models import HfTextGenerationProcessor


def get_mbart_tokenizer(
    vocab_path,
    special_input_ids=dict(),
):
    """
    Returns an MBartTokenizer initialized with the provided vocabulary file and special input IDs.

    Args:
        vocab_path (str): The path to the vocabulary file.
        special_input_ids (Dict, optional): A dictionary of special input IDs. Defaults to an empty dictionary.

    Returns:
        MBartTokenizer: The initialized MBartTokenizer.
    """
    assert os.path.exists(vocab_path)
    tokenizer = MBartTokenizer(vocab_path)
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class MBartProcessor(HfTextGenerationProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
        special_input_ids: Optional[Dict] = dict(),
    ):
        """
        Initializes an MBartProcessor with the provided parameters.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to 48.
            special_input_ids (Dict, optional): A dictionary of special input IDs. Defaults to an empty dictionary.
        """
        tokenizer = get_mbart_tokenizer(
            vocab_path,
            special_input_ids=special_input_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
