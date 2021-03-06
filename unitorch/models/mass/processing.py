# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch


from unitorch.functions import pop_first_non_none_value
from unitorch.models.bert import get_bert_tokenizer
from unitorch.models import HuggingfaceGenerationProcessor


class MASSProcessor(HuggingfaceGenerationProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 30,
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        special_tokens_ids: Optional[Dict] = dict(),
    ):
        """
        Args:
            vocab_path: vocab file path in mass tokenizer
            max_seq_length: max sequence length encode text
            max_gen_seq_length: max sequence length decode text
            do_lower_case: if do lower case to input text
            do_basic_tokenize: if do basic tokenize to input text
            special_tokens_ids: special tokens dict in mass tokenizer
        """
        tokenizer = get_bert_tokenizer(
            vocab_path,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            special_tokens_ids=special_tokens_ids,
        )
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
