# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.qwen import QWenVLProcessor as _QWenVLProcessor
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
from unitorch.cli.models.qwen import pretrained_qwen_infos


class QWenVLProcessor(_QWenVLProcessor):
    """Processor for Bloom language models."""

    def __init__(
        self,
        tokenizer_file: str,
        vision_config_path: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the BloomProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merges file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to 128.
        """
        super().__init__(
            tokenizer_file=tokenizer_file,
            vision_config_path=vision_config_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("core/process/qwen_vl")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomProcessor from the core configuration.

        Args:
            config (Config): The core configuration object.

        Returns:
            BloomProcessor: An instance of BloomProcessor initialized with the provided configuration.
        """
        config.set_default_section("core/process/qwen_vl")
        pretrained_name = config.getoption("pretrained_name", "qwen2_5-vl-3b-instruct")
        tokenizer_file = config.getoption("tokenizer_file", None)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(
                pretrained_qwen_infos, pretrained_name, "tokenizer_config"
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
                pretrained_qwen_infos, pretrained_name, "special_tokens_map"
            ),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        chat_template = config.getoption("chat_template", None)
        chat_template = pop_value(
            chat_template,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "chat_template"),
            check_none=False,
        )
        chat_template = (
            cached_path(chat_template) if chat_template is not None else None
        )

        return {
            "tokenizer_file": tokenizer_file,
            "vision_config_path": vision_config_path,
            "tokenizer_config": tokenizer_config,
            "special_tokens_map": special_tokens_map,
            "chat_template": chat_template,
        }

    @register_process("core/process/qwen_vl/chat_template")
    def _chat_template(
        self,
        messages: List[Dict[str, Any]],
    ):
        return super().chat_template(messages=messages)

    @register_process("core/process/qwen_vl/generation/inputs")
    def _generation_inputs(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
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
            images=images,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            pixel_values=outputs.pixel_values,
            image_grid_thw=outputs.image_grid_thw,
        )

    @register_process("core/process/qwen_vl/generation/labels")
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

    @register_process("core/process/qwen_vl/generation")
    def _generation(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        text_pair: str,
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
            images=images,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            image_grid_thw=outputs.image_grid_thw,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/process/qwen_vl/dpo/generation")
    def _dpo_generation(
        self,
        text: str,
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        win_text_pair: str,
        lose_text_pair: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        inputs = super().generation_inputs(
            text=text,
            images=images,
            max_seq_length=max_seq_length,
        )
        win_labels = super().generation_labels(
            text=win_text_pair,
            max_gen_seq_length=max_gen_seq_length,
        )
        lose_labels = super().generation_labels(
            text=lose_text_pair,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorsInputs(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            win_input_ids=win_labels.input_ids,
            win_attention_mask=win_labels.attention_mask,
            lose_input_ids=lose_labels.input_ids,
            lose_attention_mask=lose_labels.attention_mask,
        )

    @register_process("core/process/qwen_vl/messages/generation")
    def _messages_generation(
        self,
        messages: List[Dict[str, Any]],
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        max_seq_length: Optional[int] = None,
    ):
        outputs = super().messages_generation(
            messages=messages,
            images=images,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=outputs.pixel_values,
            image_grid_thw=outputs.image_grid_thw,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/process/qwen_vl/messages/dpo/generation")
    def _messages_dpo_generation(
        self,
        messages: List[Dict[str, Any]],
        images: Union[Image.Image, str, List[Image.Image], List[str]],
        win_messages: List[Dict[str, Any]],
        lose_messages: List[Dict[str, Any]],
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        if isinstance(messages, dict):
            messages = [messages]
        if isinstance(win_messages, dict):
            win_messages = [win_messages]
        if isinstance(lose_messages, dict):
            lose_messages = [lose_messages]
        inputs = super().generation_inputs(
            text=super().chat_template(messages=messages),
            images=images,
            max_seq_length=max_seq_length,
        )
        win_labels = super().generation_labels(
            text=super().chat_template(messages=win_messages),
            max_gen_seq_length=max_gen_seq_length,
        )
        lose_labels = super().generation_labels(
            text=super().chat_template(messages=lose_messages),
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorsInputs(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            win_input_ids=win_labels.input_ids,
            win_attention_mask=win_labels.attention_mask,
            lose_input_ids=lose_labels.input_ids,
            lose_attention_mask=lose_labels.attention_mask,
        )

    @register_process("core/postprocess/qwen_vl/detokenize")
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
                f"Unsupported type for Qwen detokenize: {type(decoded[0])}"
            )
        results["decoded"] = decoded
        return WriterOutputs(results)
