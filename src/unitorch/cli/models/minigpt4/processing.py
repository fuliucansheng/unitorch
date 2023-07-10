# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.minigpt4 import (
    MiniGPT4ViTLlamaProcessor as _MiniGPT4ViTLlamaProcessor,
)
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
from unitorch.cli.models.minigpt4 import pretrained_minigpt4_infos


class MiniGPT4ViTLlamaProcessor(_MiniGPT4ViTLlamaProcessor):
    """Processor for Llama models."""

    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the MiniGPT4ViTLlamaProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to 128.
        """
        super().__init__(
            vocab_file=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("core/process/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of MiniGPT4ViTLlamaProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            MiniGPT4ViTLlamaProcessor: An instance of MiniGPT4ViTLlamaProcessor.
        """
        config.set_default_section("core/process/minigpt4")
        pretrained_name = config.getoption("pretrained_name", "default-minigpt4")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        return {
            "vocab_path": vocab_path,
            "vision_config_path": vision_config_path,
        }

    @register_process("core/process/minigpt4/prompt")
    def _prompt(
        self,
        prefix_text: str,
        suffix_text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = None,
    ):
        """
        Process inputs for prompt-based generation.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: Processed tensors inputs.
        """
        outputs = super().prompt(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            image=image,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            prefix_input_ids=outputs.prefix_input_ids,
            suffix_input_ids=outputs.suffix_input_ids,
            pixel_values=outputs.pixel_values,
        )

    @register_process("core/process/minigpt4/generation/inputs")
    def _generation_inputs(
        self,
        prefix_text: str,
        suffix_text: str,
        image: Image.Image,
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
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            image=image,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            prefix_input_ids=outputs.prefix_input_ids,
            suffix_input_ids=outputs.suffix_input_ids,
            pixel_values=outputs.pixel_values,
        )

    @register_process("core/process/minigpt4/generation/labels")
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

    @register_process("core/process/minigpt4/generation")
    def _generation(
        self,
        prefix_text: str,
        suffix_text: str,
        text_pair: str,
        image: Image.Image,
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
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            text_pair=text_pair,
            image=image,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorsInputs(
            prefix_input_ids=outputs.prefix_input_ids,
            prefix_attention_mask=outputs.prefix_attention_mask,
            suffix_input_ids=outputs.suffix_input_ids,
            suffix_attention_mask=outputs.suffix_attention_mask,
            decoder_input_ids=outputs.input_ids_pair,
            decoder_attention_mask=outputs.attention_mask_pair,
            pixel_values=outputs.pixel_values,
        ), GenerationTargets(
            refs=outputs.input_ids_label,
            masks=outputs.attention_mask_label,
        )

    @register_process("core/postprocess/minigpt4/detokenize")
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
        results["decoded"] = decoded
        return WriterOutputs(results)
