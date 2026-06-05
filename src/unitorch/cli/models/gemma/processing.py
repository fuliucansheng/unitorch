# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
from typing import Any, Dict, List, Optional

from unitorch.models.gemma import GemmaProcessor as _GemmaProcessor
from unitorch.cli import cached_path, config_defaults_init, register_process
from unitorch.cli import WriterOutputs
from unitorch.cli.models import GenerationOutputs, GenerationTargets, TensorInputs
from unitorch.cli.models.gemma import (
    resolve_pretrained_gemma_path,
)


class GemmaProcessor(_GemmaProcessor):
    """Processor for Gemma decoder-only generation tasks."""

    def __init__(
        self,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        max_gen_seq_length: Optional[int] = 512,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @config_defaults_init("core/process/gemma")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/process/gemma")
        pretrained_name = config.getoption("pretrained_name", "gemma-4-12b")

        tokenizer_file = config.getoption("tokenizer_file", None)
        if tokenizer_file is None:
            tokenizer_file = resolve_pretrained_gemma_path(pretrained_name, "tokenizer")
        else:
            tokenizer_file = cached_path(tokenizer_file)

        tokenizer_config = config.getoption("tokenizer_config", None)
        if tokenizer_config is None:
            tokenizer_config = resolve_pretrained_gemma_path(
                pretrained_name,
                "tokenizer_config",
            )
        else:
            tokenizer_config = cached_path(tokenizer_config)

        chat_template = config.getoption("chat_template", None)
        chat_template = cached_path(chat_template) if chat_template is not None else None

        return {
            "tokenizer_file": tokenizer_file,
            "tokenizer_config": tokenizer_config,
            "chat_template": chat_template,
        }

    @register_process("core/process/gemma/chat_template")
    def _chat_template(
        self,
        messages: List[Dict[str, Any]],
    ):
        return super().chat_template(messages=messages)

    @register_process("core/process/gemma/generation/inputs")
    def _generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().generation_inputs(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    @register_process("core/process/gemma/generation/labels")
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

    @register_process("core/process/gemma/generation")
    def _generation(
        self,
        text: str,
        text_pair: str,
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
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/process/gemma/messages/generation")
    def _messages_generation(
        self,
        messages: List[Dict[str, Any]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().messages_generation(
            messages=messages,
            max_seq_length=max_seq_length,
        )
        return TensorInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/postprocess/gemma/detokenize")
    def _detokenize(
        self,
        outputs: GenerationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.sequences.shape[0]

        decoded = super().detokenize(sequences=outputs.sequences)
        cleanup_string = lambda text: re.sub(r"\n", " ", text).strip()
        if isinstance(decoded[0], list):
            decoded = [list(map(cleanup_string, sequence)) for sequence in decoded]
        elif isinstance(decoded[0], str):
            decoded = list(map(cleanup_string, decoded))
        else:
            raise ValueError(
                f"Unsupported type for Gemma detokenize: {type(decoded[0])}"
            )
        results["decoded"] = decoded
        return WriterOutputs(results)
