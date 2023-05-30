# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import XLMProphetNetTokenizer
from unitorch.models import HfTextGenerationProcessor


def get_xprophetnet_tokenizer(
    vocab_path: str,
    special_input_ids: Optional[Dict] = dict(),
):
    assert os.path.exists(vocab_path)
    tokenizer = XLMProphetNetTokenizer(vocab_path)
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class XProphetNetProcessor(HfTextGenerationProcessor):
    def __init__(
        self,
        vocab_path: str,
        special_input_ids: Optional[Dict] = dict(),
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        tokenizer = get_xprophetnet_tokenizer(
            vocab_path,
            special_input_ids=special_input_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
