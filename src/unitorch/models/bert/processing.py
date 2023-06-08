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
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import HfTextClassificationProcessor, GenericOutputs


def get_random_word(vocab_words: List[str]) -> str:
    i = random.randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def get_random_mask_indexes(
    tokens: List[str],
    masked_lm_prob: Optional[float] = 0.15,
    do_whole_word_mask: Optional[bool] = True,
    max_predictions_per_seq: Optional[int] = 20,
    special_tokens: List[str] = [],
) -> List[int]:
    cand_indexes = []
    for i, token in enumerate(tokens):
        if token in special_tokens:
            continue
        if (
            do_whole_word_mask
            and len(cand_indexes) >= 1
            and token.startswith("##")
            and cand_indexes[-1][-1] == i - 1
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    random.shuffle(cand_indexes)
    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(covered_indexes) >= num_to_predict:
            break
        if len(covered_indexes) + len(index_set) > num_to_predict or any(
            i in covered_indexes for i in index_set
        ):
            continue
        covered_indexes.update(index_set)
    return list(covered_indexes)


def get_bert_tokenizer(
    vocab_path,
    do_lower_case: Optional[bool] = True,
    do_basic_tokenize: Optional[bool] = True,
    special_input_ids: Optional[Dict] = dict(),
):
    assert os.path.exists(vocab_path)
    tokenizer = BertTokenizer(
        vocab_path,
        do_lower_case=do_lower_case,
        do_basic_tokenize=do_basic_tokenize,
    )
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class BertProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        special_input_ids: Optional[Dict] = dict(),
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        do_whole_word_mask: Optional[bool] = True,
        masked_lm_prob: Optional[float] = 0.15,
        max_predictions_per_seq: Optional[int] = 20,
    ):
        """
        Initializes the BertProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (Optional[int], optional): The maximum sequence length. Defaults to 128.
            special_input_ids (Optional[Dict], optional): Special input IDs mapping. Defaults to an empty dictionary.
            do_lower_case (Optional[bool], optional): Whether to perform lowercase tokenization. Defaults to True.
            do_basic_tokenize (Optional[bool], optional): Whether to perform basic tokenization. Defaults to True.
            do_whole_word_mask (Optional[bool], optional): Whether to perform whole word masking. Defaults to True.
            masked_lm_prob (Optional[float], optional): The probability of masking a token for pretraining. Defaults to 0.15.
            max_predictions_per_seq (Optional[int], optional): The maximum number of masked tokens per sequence for pretraining. Defaults to 20.
        """
        tokenizer = get_bert_tokenizer(
            vocab_path,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            special_input_ids=special_input_ids,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        self.do_whole_word_mask = do_whole_word_mask
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(self.tokenizer.vocab.keys())

    def pretrain(
        self,
        text: str,
        text_pair: str,
        nsp_label: int,
        max_seq_length: Optional[int] = None,
        masked_lm_prob: Optional[float] = None,
        do_whole_word_mask: Optional[bool] = None,
        max_predictions_per_seq: Optional[int] = None,
    ):
        """
        The Bert pretrain processor on the given text and text pair.

        Args:
            text (str): The input text.
            text_pair (str): The input text pair.
            nsp_label (int): The next sentence prediction label.
            max_seq_length (Optional[int], optional): The maximum sequence length. Defaults to None.
            masked_lm_prob (Optional[float], optional): The probability of masking a token for pretraining. Defaults to None.
            do_whole_word_mask (Optional[bool], optional): Whether to perform whole word masking. Defaults to None.
            max_predictions_per_seq (Optional[int], optional): The maximum number of masked tokens per sequence for pretraining. Defaults to None.

        Returns:
            GenericOutputs: pretrain processing outputs.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )

        masked_lm_prob = pop_value(
            masked_lm_prob,
            self.masked_lm_prob,
        )

        do_whole_word_mask = pop_value(
            do_whole_word_mask,
            self.do_whole_word_mask,
        )

        max_predictions_per_seq = pop_value(
            max_predictions_per_seq,
            self.max_predictions_per_seq,
        )

        _tokens = self.tokenizer.tokenize(str(text))
        tokens_pair = self.tokenizer.tokenize(str(text_pair))
        truncate_sequence_pair(_tokens, tokens_pair, max_seq_length - 3)
        tokens = (
            [self.cls_token]
            + _tokens
            + [self.sep_token]
            + tokens_pair
            + [self.sep_token]
        )

        covered_indexes = get_random_mask_indexes(
            tokens,
            masked_lm_prob,
            do_whole_word_mask,
            max_predictions_per_seq,
            special_tokens=[self.cls_token, self.sep_token],
        )
        label = [
            tokens[pos] if pos in covered_indexes else self.pad_token
            for pos in range(max_seq_length)
        ]
        label_mask = [
            1 if pos in covered_indexes else 0 for pos in range(max_seq_length)
        ]
        label = self.tokenizer.convert_tokens_to_ids(label)

        for index in covered_indexes:
            mask_token = None
            if random.random() < 0.8:
                mask_token = self.mask_token
            else:
                mask_token = (
                    tokens[index]
                    if random.random() < 0.5
                    else get_random_word(self.vocab_words)
                )
            tokens[index] = mask_token

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] + [0] * len(_tokens) + [0] + [1] * len(tokens_pair) + [1]
        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += len(padding) * [self.pad_token_id]
        attention_mask += padding
        token_type_ids += len(padding) * [1]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            position_ids=torch.tensor(list(range(max_seq_length)), dtype=torch.long),
            nsp_label=torch.tensor(int(nsp_label), dtype=torch.long),
            mlm_label=torch.tensor(label, dtype=torch.long),
            mlm_label_mask=torch.tensor(label_mask, dtype=torch.long),
        )
