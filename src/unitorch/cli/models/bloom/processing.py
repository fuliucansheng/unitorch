# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bloom import BloomProcessor as _BloomProcessor
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
from unitorch.cli.models.bloom import pretrained_bloom_infos


class BloomProcessor(_BloomProcessor):
    """Processor for Bloom language models."""

    def __init__(
        self,
        tokenizer_file: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the BloomProcessor.

        Args:
            tokenizer_file (str): The path to the tokenizer file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to 128.
        """
        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("core/process/bloom")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomProcessor from the core configuration.

        Args:
            config (Config): The core configuration object.

        Returns:
            BloomProcessor: An instance of BloomProcessor initialized with the provided configuration.
        """
        config.set_default_section("core/process/bloom")
        pretrained_name = config.getoption("pretrained_name", "bloom-560m")
        tokenizer_file = config.getoption("tokenizer_file", None)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(
                pretrained_bloom_infos, pretrained_name, "tokenizer_config"
            ),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config) if tokenizer_config is not None else None
        )

        special_tokens_map = config.getoption("special_tokens_map", None)
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(
                pretrained_bloom_infos, pretrained_name, "special_tokens_map"
            ),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        chat_template = config.getoption("chat_template", None)
        chat_template = pop_value(
            chat_template,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "chat_template"),
            check_none=False,
        )
        chat_template = (
            cached_path(chat_template) if chat_template is not None else None
        )

        return {
            "tokenizer_file": tokenizer_file,
            "tokenizer_config": tokenizer_config,
            "special_tokens_map": special_tokens_map,
            "chat_template": chat_template,
        }

    @register_process("core/process/bloom/chat_template")
    def _chat_template(
        self,
        messages: List[Dict[str, Any]],
    ):
        return super().chat_template(messages=messages)

    @register_process("core/process/bloom/classification")
    def _classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Perform classification using the Bloom model.

        Args:
            text (str): The input text.
            text_pair (str, optional): The second input text for pair-based classification. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The input tensors for classification.
        """
        outputs = super().classification(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    @register_process("core/process/bloom/generation/inputs")
    def _generation_inputs(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Preprocess the input text for generation tasks.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The processed input tensors.
        """
        outputs = super().generation_inputs(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(input_ids=outputs.input_ids)

    @register_process("core/process/bloom/generation/labels")
    def _generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Preprocess the target text for generation tasks.

        Args:
            text (str): The target text.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to None.

        Returns:
            GenerationTargets: The processed generation targets.
        """
        outputs = super().generation_labels(
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.input_ids,
            masks=outputs.attention_mask,
        )

    @register_process("core/process/bloom/generation")
    def _generation(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Preprocess the input and target texts for generation tasks.

        Args:
            text (str): The input text.
            text_pair (str, optional): The paired input text. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to None.

        Returns:
            Tuple[TensorsInputs, GenerationTargets]: The processed input tensors and generation targets.
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
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/process/bloom/messages/generation")
    def _messages_generation(
        self,
        messages: List[Dict[str, Any]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().messages_generation(
            messages=messages,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/postprocess/bloom/detokenize")
    def _detokenize(
        self,
        outputs: GenerationOutputs,
    ):
        """
        Detokenize the generated sequences.

        Args:
            outputs (GenerationOutputs): The generation outputs.

        Returns:
            WriterOutputs: The detokenized writer outputs.
        """
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.sequences.shape[0]

        decoded = super().detokenize(sequences=outputs.sequences)
        cleanup_string = lambda text: re.sub(r"\n", " ", text)
        if isinstance(decoded[0], list):
            decoded = [list(map(cleanup_string, sequence)) for sequence in decoded]
        elif isinstance(decoded[0], str):
            decoded = list(map(cleanup_string, decoded))
        else:
            raise ValueError(
                f"Unsupported type for Bloom detokenize: {type(decoded[0])}"
            )
        results["decoded"] = decoded
        return WriterOutputs(results)
