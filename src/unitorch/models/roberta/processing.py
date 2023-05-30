# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import partial

import torch
from transformers import RobertaTokenizer
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import GenericOutputs, HfTextClassificationProcessor


def get_roberta_tokenizer(vocab_path, merge_path):
    """
    Retrieves the RobertaTokenizer with the specified vocabulary and merge files.

    Args:
        vocab_path (str): The path to the vocabulary file.
        merge_path (str): The path to the merge file.

    Returns:
        RobertaTokenizer: The RobertaTokenizer instance.
    """
    assert os.path.exists(vocab_path) and os.path.exists(merge_path)
    tokenizer = RobertaTokenizer(vocab_path, merge_path)
    return tokenizer


class RobertaProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 0,
    ):
        """
        Initializes a RobertaProcessor for text classification tasks.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merge file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            source_type_id (int, optional): The ID for the source type. Defaults to 0.
            target_type_id (int, optional): The ID for the target type. Defaults to 0.
        """
        tokenizer = get_roberta_tokenizer(
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
