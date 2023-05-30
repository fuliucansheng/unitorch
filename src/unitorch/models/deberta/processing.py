# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import partial

import torch
from transformers import DebertaTokenizer
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import GenericOutputs, HfTextClassificationProcessor


def get_deberta_tokenizer(vocab_path, merge_path):
    assert os.path.exists(vocab_path) and os.path.exists(merge_path)
    tokenizer = DebertaTokenizer(vocab_path, merge_path)
    return tokenizer


class DebertaProcessor(HfTextClassificationProcessor):
    """
    Processor for DeBERTa-based text classification models.
    """

    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 0,
    ):
        """
        Initializes the DebertaProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            merge_path (str): Path to the merge file.
            max_seq_length (Optional[int]): Maximum sequence length. Defaults to 128.
            source_type_id (Optional[int]): Source type ID for token types. Defaults to 0.
            target_type_id (Optional[int]): Target type ID for token types. Defaults to 0.
        """
        tokenizer = get_deberta_tokenizer(
            vocab_path,
            merge_path,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            position_start_id=tokenizer.pad_token_id + 1,
        )
