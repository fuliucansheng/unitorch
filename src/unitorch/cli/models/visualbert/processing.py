# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.visualbert import VisualBertProcessor as _VisualBertProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.visualbert import pretrained_visualbert_infos


class VisualBertProcessor(_VisualBertProcessor):
    """VisualBERT Processor for text and image inputs."""

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
        Initialize the VisualBertProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to 128.
            special_input_ids (Optional[Dict]): A dictionary containing special input IDs. Defaults to an empty dictionary.
            do_lower_case (Optional[bool]): Whether to convert the text to lowercase. Defaults to True.
            do_basic_tokenize (Optional[bool]): Whether to perform basic tokenization. Defaults to True.
            do_whole_word_mask (Optional[bool]): Whether to use whole-word masking. Defaults to True.
            masked_lm_prob (Optional[float]): The probability of masked language model masking. Defaults to 0.15.
            max_predictions_per_seq (Optional[int]): The maximum number of masked language model predictions per sequence. Defaults to 20.
        """
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            special_input_ids=special_input_ids,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            do_whole_word_mask=do_whole_word_mask,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq,
        )

    @classmethod
    @add_default_section_for_init("core/process/visualbert")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of VisualBertProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the processor's configuration.
        """
        config.set_default_section("core/process/visualbert")
        pretrained_name = config.getoption("pretrained_name", "default-visualbert")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_visualbert_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("core/process/visualbert/classification")
    def _classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Perform classification processing on the input text.

        Args:
            text (str): The input text.
            text_pair (Optional[str]): The second input text for sentence pair classification. Defaults to None.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The processed tensors as inputs to the model.
        """
        outputs = super().classification(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            token_type_ids=outputs.token_type_ids,
            position_ids=outputs.position_ids,
        )

    @register_process("core/process/visualbert/pretrain")
    def _pretrain(
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
        Perform pretraining processing on the input text and labels.

        Args:
            text (str): The input text.
            text_pair (str): The second input text for sentence pair pretraining.
            nsp_label (int): The next sentence prediction label.
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to None.
            masked_lm_prob (Optional[float]): The probability of masked language model masking. Defaults to None.
            do_whole_word_mask (Optional[bool]): Whether to use whole-word masking. Defaults to None.
            max_predictions_per_seq (Optional[int]): The maximum number of masked language model predictions per sequence. Defaults to None.

        Returns:
            TensorsInputs: The processed tensors as inputs to the model.
        """
        outputs = super().pretrain(
            text=text,
            text_pair=text_pair,
            nsp_label=nsp_label,
            max_seq_length=max_seq_length,
            masked_lm_prob=masked_lm_prob,
            do_whole_word_mask=do_whole_word_mask,
            max_predictions_per_seq=max_predictions_per_seq,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            token_type_ids=outputs.token_type_ids,
            position_ids=outputs.position_ids,
            nsp_label=outputs.nsp_label,
            mlm_label=outputs.mlm_label,
            mlm_label_mask=outputs.mlm_label_mask,
        )
