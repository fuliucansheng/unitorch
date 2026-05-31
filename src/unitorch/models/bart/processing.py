# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Dict, Optional

from transformers import BartTokenizer

from unitorch.models import HfTextGenerationProcessor


def get_bart_tokenizer(
    vocab_path: str,
    merge_path: str,
    special_input_ids: Optional[Dict] = None,
) -> BartTokenizer:
    assert os.path.exists(vocab_path) and os.path.exists(merge_path)
    tokenizer = BartTokenizer(vocab_path, merge_path)
    for token, token_id in (special_input_ids or {}).items():
        tokenizer.added_tokens_encoder[token] = token_id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[token_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class BartProcessor(HfTextGenerationProcessor):
    """Text generation processor for BART models."""

    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        special_input_ids: Optional[Dict] = None,
        max_seq_length: int = 128,
        max_gen_seq_length: int = 48,
    ) -> None:
        super().__init__(
            tokenizer=get_bart_tokenizer(vocab_path, merge_path, special_input_ids),
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
