# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import random
from typing import Dict, List, Optional

import torch
from transformers import BertTokenizer

from unitorch.models import HfTextClassificationProcessor, GenericOutputs
from unitorch.utils import pop_value, truncate_sequence_pair


def get_random_word(vocab_words: List[str]) -> str:
    return vocab_words[random.randint(0, len(vocab_words) - 1)]


def get_random_mask_indexes(
    tokens: List[str],
    masked_lm_prob: float = 0.15,
    do_whole_word_mask: bool = True,
    max_predictions_per_seq: int = 20,
    special_tokens: Optional[List[str]] = None,
) -> List[int]:
    """Return a list of token indices selected for MLM masking."""
    special_tokens = special_tokens or []
    cand_indexes: List[List[int]] = []
    for i, token in enumerate(tokens):
        if token in special_tokens:
            continue
        if (
            do_whole_word_mask
            and cand_indexes
            and token.startswith("##")
            and cand_indexes[-1][-1] == i - 1
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions_per_seq, max(1, round(len(tokens) * masked_lm_prob)))
    covered: set = set()
    for index_set in cand_indexes:
        if len(covered) >= num_to_predict:
            break
        if len(covered) + len(index_set) > num_to_predict or any(i in covered for i in index_set):
            continue
        covered.update(index_set)
    return list(covered)


def get_bert_tokenizer(
    vocab_path: str,
    do_lower_case: bool = True,
    do_basic_tokenize: bool = True,
    special_input_ids: Optional[Dict] = None,
) -> BertTokenizer:
    assert os.path.exists(vocab_path)
    tokenizer = BertTokenizer(vocab_path, do_lower_case=do_lower_case, do_basic_tokenize=do_basic_tokenize)
    for token, token_id in (special_input_ids or {}).items():
        tokenizer.added_tokens_encoder[token] = token_id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[token_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer


class BertProcessor(HfTextClassificationProcessor):
    """Text processor for BERT models, including MLM pre-training support."""

    def __init__(
        self,
        vocab_path: str,
        max_seq_length: int = 128,
        special_input_ids: Optional[Dict] = None,
        do_lower_case: bool = True,
        do_basic_tokenize: bool = True,
        do_whole_word_mask: bool = True,
        masked_lm_prob: float = 0.15,
        max_predictions_per_seq: int = 20,
    ) -> None:
        super().__init__(
            tokenizer=get_bert_tokenizer(
                vocab_path,
                do_lower_case=do_lower_case,
                do_basic_tokenize=do_basic_tokenize,
                special_input_ids=special_input_ids,
            ),
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
    ) -> GenericOutputs:
        """Process a sentence pair for BERT pre-training (MLM + NSP)."""
        max_seq_length = pop_value(max_seq_length, self.max_seq_length)
        masked_lm_prob = pop_value(masked_lm_prob, self.masked_lm_prob)
        do_whole_word_mask = pop_value(do_whole_word_mask, self.do_whole_word_mask)
        max_predictions_per_seq = pop_value(max_predictions_per_seq, self.max_predictions_per_seq)

        tokens_a = self.tokenizer.tokenize(str(text))
        tokens_b = self.tokenizer.tokenize(str(text_pair))
        truncate_sequence_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = [self.cls_token] + tokens_a + [self.sep_token] + tokens_b + [self.sep_token]

        covered_indexes = get_random_mask_indexes(
            tokens,
            masked_lm_prob=masked_lm_prob,
            do_whole_word_mask=do_whole_word_mask,
            max_predictions_per_seq=max_predictions_per_seq,
            special_tokens=[self.cls_token, self.sep_token],
        )

        mlm_label = [
            tokens[i] if i in covered_indexes else self.pad_token
            for i in range(max_seq_length)
        ]
        mlm_label_mask = [1 if i in covered_indexes else 0 for i in range(max_seq_length)]
        mlm_label = self.tokenizer.convert_tokens_to_ids(mlm_label)

        for idx in covered_indexes:
            if random.random() < 0.8:
                tokens[idx] = self.mask_token
            elif random.random() < 0.5:
                pass  # keep original token
            else:
                tokens[idx] = get_random_word(self.vocab_words)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        attention_mask = [1] * len(input_ids)

        pad_len = max_seq_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [1] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            position_ids=torch.arange(max_seq_length, dtype=torch.long),
            nsp_label=torch.tensor(int(nsp_label), dtype=torch.long),
            mlm_label=torch.tensor(mlm_label, dtype=torch.long),
            mlm_label_mask=torch.tensor(mlm_label_mask, dtype=torch.long),
        )
