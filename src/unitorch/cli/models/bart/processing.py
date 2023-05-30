# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bart import BartProcessor as _BartProcessor
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
from unitorch.cli.models.bart import pretrained_bart_infos


class BartProcessor(_BartProcessor):
    """Class for processing data with BART model."""

    def __init__(
        self,
        vocab_path: str,
        merge_path: str,
        special_input_ids: Optional[Dict] = dict(),
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
    ):
        """
        Initialize BartProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merge file.
            special_input_ids (Dict, optional): Special input IDs. Defaults to an empty dictionary.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to 48.
        """
        super().__init__(
            vocab_path=vocab_path,
            merge_path=merge_path,
            special_input_ids=special_input_ids,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("core/process/bart")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BartProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            BartProcessor: An instance of BartProcessor.
        """
        config.set_default_section("core/process/bart")
        pretrained_name = config.getoption("pretrained_name", "default-bart")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_bart_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_bart_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        return {
            "vocab_path": vocab_path,
            "merge_path": merge_path,
        }

    @register_process("core/process/bart/generation")
    def _generation(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Perform generation with BART model.

        Args:
            text (str): The input text.
            text_pair (str, optional): The paired text. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to None.

        Returns:
            TensorsInputs, GenerationTargets: The input tensors and generation targets.
        """
        outputs = super().generation(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            decoder_input_ids=outputs.input_ids_pair,
            decoder_attention_mask=outputs.attention_mask_pair,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/process/bart/generation/inputs")
    def _generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Prepare input tensors for generation with BART model.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The input tensors.
        """
        outputs = super().generation_inputs(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(input_ids=outputs.input_ids)

    @register_process("core/process/bart/generation/labels")
    def _generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Prepare generation targets for BART model.

        Args:
            text (str): The input text.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to None.

        Returns:
            GenerationTargets: The generation targets.
        """
        outputs = super().generation_labels(
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.input_ids,
            masks=outputs.attention_mask,
        )

    @register_process("core/postprocess/bart/detokenize")
    def _detokenize(
        self,
        outputs: GenerationOutputs,
    ):
        """
        Detokenize the generation outputs.

        Args:
            outputs (GenerationOutputs): The generation outputs.

        Returns:
            WriterOutputs: The detokenized results.
        """
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.sequences.shape[0]

        decoded = super().detokenize(sequences=outputs.sequences)
        results["decoded"] = decoded
        return WriterOutputs(results)
