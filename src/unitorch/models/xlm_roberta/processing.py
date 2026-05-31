# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Optional
from transformers import XLMRobertaTokenizer
from unitorch.models import GenericOutputs, HfTextClassificationProcessor


def get_xlm_roberta_tokenizer(vocab_path: str) -> XLMRobertaTokenizer:
    """
    Creates an XLMRobertaTokenizer from a vocabulary file.

    Args:
        vocab_path (str): Path to the vocabulary file.

    Returns:
        XLMRobertaTokenizer: Configured tokenizer.
    """
    assert os.path.exists(vocab_path)
    return XLMRobertaTokenizer(vocab_path)


class XLMRobertaProcessor(HfTextClassificationProcessor):
    """
    Processor for XLM-RoBERTa text classification models.
    """

    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 0,
    ):
        """
        Initializes the XLMRobertaProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 128.
            source_type_id (int, optional): Source type ID. Defaults to 0.
            target_type_id (int, optional): Target type ID. Defaults to 0.
        """
        tokenizer = get_xlm_roberta_tokenizer(vocab_path)
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            position_start_id=self.pad_token_id + 1,
        )
