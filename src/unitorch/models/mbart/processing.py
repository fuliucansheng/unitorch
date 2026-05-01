# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Dict, Optional

import torch
from transformers import MBartTokenizer

from unitorch.utils import pop_value
from unitorch.models import HfTextGenerationProcessor


def get_mbart_tokenizer(
    vocab_path: str,
    special_input_ids: Dict = {},
):
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
        special_input_ids: Optional[Dict] = {},
    ):
        tokenizer = get_mbart_tokenizer(
            vocab_path,
            special_input_ids=special_input_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
