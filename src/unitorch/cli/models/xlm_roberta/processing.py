# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.xlm_roberta import XLMRobertaProcessor as _XLMRobertaProcessor
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
from unitorch.cli.models.xlm_roberta import pretrained_xlm_roberta_infos


class XLMRobertaProcessor(_XLMRobertaProcessor):
    """XLM-RoBERTa Processor for handling text processing tasks."""

    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 0,
    ):
        """
        Initialize XLMRobertaProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            source_type_id (int, optional): The source type ID. Defaults to 0.
            target_type_id (int, optional): The target type ID. Defaults to 0.
        """
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
        )

    @classmethod
    @add_default_section_for_init("core/process/xlm_roberta")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of XLMRobertaProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The processed arguments for initializing the processor.
        """
        config.set_default_section("core/process/xlm_roberta")
        pretrained_name = config.getoption("pretrained_name", "default-xlm-roberta")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_xlm_roberta_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("core/process/xlm_roberta/classification")
    def _classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Process text for classification tasks.

        Args:
            text (str): The input text.
            text_pair (str, optional): The paired text. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The processed input tensors.
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
