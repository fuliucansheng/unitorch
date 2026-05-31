# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Optional
from transformers import RobertaTokenizer
from unitorch.models import GenericOutputs, HfTextClassificationProcessor


def get_roberta_tokenizer(vocab_path: str, merge_path: str) -> RobertaTokenizer:
    """
    Creates a RobertaTokenizer from vocabulary and merge files.

    Args:
        vocab_path (str): Path to the vocabulary file.
        merge_path (str): Path to the merge file.

    Returns:
        RobertaTokenizer: Configured tokenizer.
    """
    assert os.path.exists(vocab_path) and os.path.exists(merge_path)
    return RobertaTokenizer(vocab_path, merge_path)


class RobertaProcessor(HfTextClassificationProcessor):
    """
    Processor for Roberta text classification models.
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
        Initializes the RobertaProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            merge_path (str): Path to the merge file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 128.
            source_type_id (int, optional): Source type ID. Defaults to 0.
            target_type_id (int, optional): Target type ID. Defaults to 0.
        """
        tokenizer = get_roberta_tokenizer(vocab_path, merge_path)
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            position_start_id=tokenizer.pad_token_id + 1,
        )
