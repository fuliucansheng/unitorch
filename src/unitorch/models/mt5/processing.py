# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers import MT5Tokenizer
from unitorch.models import HfTextGenerationProcessor


def get_mt5_tokenizer(
    vocab_path,
    special_input_ids=dict(),
):
    """
    Creates and returns an instance of the MT5 tokenizer.

    Args:
        vocab_path (str): The path to the vocabulary file.
        special_input_ids (Dict, optional): Dictionary of special input tokens and their corresponding IDs. Defaults to an empty dictionary.

    Returns:
        MT5Tokenizer: The MT5 tokenizer.
    """
    tokenizer = MT5Tokenizer(vocab_path)
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class MT5Processor(HfTextGenerationProcessor):
    def __init__(
        self,
        vocab_path: str,
        special_input_ids: Optional[Dict] = dict(),
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initializes an MT5Processor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            special_input_ids (Dict, optional): Dictionary of special input tokens and their corresponding IDs. Defaults to an empty dictionary.
            max_seq_length (int, optional): The maximum length of input sequences. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum length of generated sequences. Defaults to 48.
        """
        tokenizer = get_mt5_tokenizer(
            vocab_path,
            special_input_ids=special_input_ids,
        )
        tokenizer.bos_token_id = 0
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(0)
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.sep_token_id = tokenizer.eos_token_id
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
