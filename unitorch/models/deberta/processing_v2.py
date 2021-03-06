# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import partial

import torch
from transformers import DebertaV2Tokenizer
from unitorch.functions import pop_first_non_none_value
from unitorch.models import GenericOutputs, _truncate_seq_pair
from unitorch.models import HuggingfaceClassificationProcessor


def _get_deberta_v2_tokenizer(vocab_path):
    assert os.path.exists(vocab_path)
    tokenizer = DebertaV2Tokenizer(vocab_path)
    return tokenizer


class DebertaV2Processor(HuggingfaceClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 1,
    ):
        """
        Args:
            vocab_path: vocab file path in deberta v2 tokenizer
            max_seq_length: max sequence length input text
            source_type_id: token type id to text_a
            target_type_id: token type id to text_b
        """
        tokenizer = _get_deberta_v2_tokenizer(
            vocab_path,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.position_start_id = self.pad_token_id + 1

    def processing_classification(
        self,
        text: str,
        text_pair: str = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Args:
            text: encode text
            text_pair: decode text
            max_seq_length: max sequence length to encode text
        """
        max_seq_length = int(
            pop_first_non_none_value(
                max_seq_length,
                self.max_seq_length,
            )
        )

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 2]
            tokens = [self.cls_token] + tokens + [self.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [self.source_type_id] * len(tokens_ids)
            tokens_mask = [1] * len(tokens_ids)
        else:
            tokens_b = self.tokenizer.tokenize(str(text_pair))
            _truncate_seq_pair(tokens, tokens_b, max_seq_length - 3)
            segment_ids = (
                [self.source_type_id]
                + [self.source_type_id] * len(tokens)
                + [self.source_type_id]
                + [self.target_type_id] * len(tokens_b)
                + [self.target_type_id]
            )
            tokens = [self.cls_token] + tokens + [self.sep_token] + tokens_b + [self.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens_mask = [1] * len(tokens_ids)

        padding = [0] * (max_seq_length - len(tokens_ids))
        tokens_ids += len(padding) * [self.pad_token_id]
        tokens_mask += padding
        segment_ids += len(padding) * [self.target_type_id]

        assert len(tokens_ids) == max_seq_length
        assert len(tokens_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return GenericOutputs(
            tokens_ids=torch.tensor(tokens_ids, dtype=torch.long),
            seg_ids=torch.tensor(segment_ids, dtype=torch.long),
            attn_mask=torch.tensor(tokens_mask, dtype=torch.long),
            pos_ids=torch.tensor(
                list(
                    range(
                        self.position_start_id,
                        self.position_start_id + max_seq_length,
                    )
                ),
                dtype=torch.long,
            ),
        )
