# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Dict, Optional
from unitorch.models.bert import BertProcessor


class VisualBertProcessor(BertProcessor):
    """
    Processor for VisualBERT-based models.
    """

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
        Initializes the VisualBertProcessor.

        Args:
            vocab_path (str): Path to the vocabulary file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 128.
            special_input_ids (Dict, optional): Special input token IDs. Defaults to empty dict.
            do_lower_case (bool, optional): Whether to lowercase text. Defaults to True.
            do_basic_tokenize (bool, optional): Whether to perform basic tokenization. Defaults to True.
            do_whole_word_mask (bool, optional): Whether to use whole-word masking. Defaults to True.
            masked_lm_prob (float, optional): Probability for masked LM. Defaults to 0.15.
            max_predictions_per_seq (int, optional): Max masked LM predictions per sequence. Defaults to 20.
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
