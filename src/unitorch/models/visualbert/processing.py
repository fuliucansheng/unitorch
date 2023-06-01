# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from random import randint, shuffle, choice

import numpy as np
import torch
from transformers import BertTokenizer

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value
from unitorch.models.bert import BertProcessor, get_bert_tokenizer


class VisualBertProcessor(BertProcessor):
    """
    Processor for VisualBERT-based models.
    """

    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        special_input_ids: Optional[Dict] = dict(),
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        do_whole_word_mask: Optional[bool] = True,
        masked_lm_prob: Optional[float] = 0.15,
        max_predictions_per_seq: Optional[int] = 20,
    ):
        """
        Initializes the VisualBertProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            max_seq_length (Optional[int]): Maximum sequence length. Defaults to 128.
            special_input_ids (Optional[Dict]): Special input IDs. Defaults to an empty dictionary.
            do_lower_case (Optional[bool]): Whether to convert text to lowercase. Defaults to True.
            do_basic_tokenize (Optional[bool]): Whether to perform basic tokenization. Defaults to True.
            do_whole_word_mask (Optional[bool]): Whether to use whole word masking. Defaults to True.
            masked_lm_prob (Optional[float]): Probability for masked LM. Defaults to 0.15.
            max_predictions_per_seq (Optional[int]): Maximum number of masked LM predictions per sequence. Defaults to 20.
        """
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            do_whole_word_mask=do_whole_word_mask,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq,
        )
