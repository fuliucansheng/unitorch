# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers import BartTokenizer
from unitorch.models import HfTextGenerationProcessor


def get_bart_tokenizer(
    vocab_path: str, merge_path: str, special_input_ids: Optional[Dict] = dict()
) -> BartTokenizer:
    assert os.path.exists(vocab_path) and os.path.exists(merge_path)
    tokenizer = BartTokenizer(vocab_path, merge_path)

    # Add special tokens to the tokenizer
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)

    return tokenizer


class BartProcessor(HfTextGenerationProcessor):
    """
    Processor for BART model.
    Inherits from HfTextGenerationProcessor.
    """

    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        special_input_ids: Optional[Dict] = dict(),
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initializes the BartProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            merge_path (str): Path to the BPE merge file.
            special_input_ids (Optional[Dict]): Optional dictionary of special input IDs.
            max_seq_length (Optional[int]): Maximum sequence length.
            max_gen_seq_length (Optional[int]): Maximum generation sequence length.
        """
        tokenizer = get_bart_tokenizer(
            vocab_path,
            merge_path,
            special_input_ids=special_input_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
