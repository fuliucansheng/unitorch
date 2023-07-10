# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.minigpt4 import (
    MiniGPT4ViTLlamaForGeneration as _MiniGPT4ViTLlamaForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.minigpt4 import pretrained_minigpt4_infos


@register_model("core/model/generation/llama", generation_model_decorator)
class MiniGPT4ViTLlamaForGeneration(_MiniGPT4ViTLlamaForGeneration):
    """MiniGPT4 model for generation tasks."""

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            blip2_config_path=blip2_config_path,
            llama_config_path=llama_config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of MiniGPT4ViTLlamaForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlamaForGeneration: An instance of LlamaForGeneration.
        """
        config.set_default_section("core/model/generation/minigpt4")
        pretrained_name = config.getoption("pretrained_name", "default-minigpt4")

        blip2_config_path = config.getoption("blip2_config_path", None)
        blip2_config_path = pop_value(
            blip2_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "blip2_config_path"
            ),
        )
        blip2_config_path = cached_path(blip2_config_path)

        llama_config_path = config.getoption("llama_config_path", None)
        llama_config_path = pop_value(
            llama_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "llama_config_path"
            ),
        )
        llama_config_path = cached_path(llama_config_path)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(blip2_config_path, llama_config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast()
    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass on the MiniGPT4ViTLlamaForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input tensor containing the input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor. Defaults to None.

        Returns:
            GenerationOutputs: The output of the generation model.
        """
        outputs = super().forward(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            decoder_input_ids=decoder_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/minigpt4")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[int] = 2,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 48,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        """
        Generate sequences using the MinGPT4 model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 0.
            decoder_end_token_id (int, optional): Decoder end token ID. Defaults to 1.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum generation sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum generation sequence length. Defaults to 48.
            repetition_penalty (float, optional): Repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to prevent repetition. Defaults to 0.
            early_stopping (bool, optional): Whether to perform early stopping. Defaults to True.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().generate(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return GenerationOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )
