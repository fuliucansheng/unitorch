# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.mbart import MBartProcessor as _MBartProcessor
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.mbart import pretrained_mbart_infos


class MBartProcessor(_MBartProcessor):
    """Processor for the MBart model."""

    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 48,
        special_input_ids: Optional[Dict] = dict(),
    ):
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            special_input_ids=special_input_ids,
        )

    @classmethod
    @add_default_section_for_init("core/process/mbart")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/process/mbart")
        pretrained_name = config.getoption("pretrained_name", "mbart-large-cc25")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_mbart_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("core/process/mbart/generation/inputs")
    def _generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().generation_inputs(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(input_ids=outputs.input_ids)

    @register_process("core/process/mbart/generation/labels")
    def _generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        outputs = super().generation_labels(
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.input_ids,
            masks=outputs.attention_mask,
        )

    @register_process("core/process/mbart/generation")
    def _generation(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        outputs = super().generation(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            decoder_input_ids=outputs.input_ids_pair,
            decoder_attention_mask=outputs.attention_mask_pair,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/postprocess/mbart/detokenize")
    def _detokenize(
        self,
        outputs: GenerationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.sequences.shape[0]

        decoded = super().detokenize(sequences=outputs.sequences)
        results["decoded"] = decoded
        return WriterOutputs(results)
