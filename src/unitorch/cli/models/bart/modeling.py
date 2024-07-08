# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bart import BartForGeneration as _BartForGeneration
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import GenerationOutputs
from unitorch.cli.models.bart import pretrained_bart_infos


@register_model("core/model/generation/bart", generation_model_decorator)
class BartForGeneration(_BartForGeneration):
    """BART model for generation tasks."""

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the BartForGeneration model.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory optimization.
        """
        super().__init__(
            config_path=config_path, gradient_checkpointing=gradient_checkpointing
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/bart")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BartForGeneration from core configuration.

        Args:
            config: The core configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            BartForGeneration: The initialized BartForGeneration instance.
        """
        config.set_default_section("core/model/generation/bart")
        pretrained_name = config.getoption("pretrained_name", "default-bart")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bart_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bart_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        """
        Forward pass of the BartForGeneration model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention mask.
            decoder_input_ids (torch.Tensor): Decoder input IDs.
            decoder_attention_mask (torch.Tensor): Decoder attention mask.

        Returns:
            GenerationOutputs: The generated sequences and their scores.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/bart")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 2,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 2,
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
        Generate sequences using the BartForGeneration model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            num_beams (int, optional): Number of beams for beam search.
            decoder_start_token_id (int, optional): ID of the decoder start token.
            decoder_end_token_id (int or List[int], optional): ID of the decoder end token.
            num_return_sequences (int, optional): Number of generated sequences to return.
            min_gen_seq_length (int, optional): Minimum length of generated sequences.
            max_gen_seq_length (int, optional): Maximum length of generated sequences.
            repetition_penalty (float, optional): Repetition penalty.
            no_repeat_ngram_size (int, optional): Size of n-grams to avoid repeating.
            early_stopping (bool, optional): Whether to stop generation early.
            length_penalty (float, optional): Length penalty for generated sequences.
            num_beam_groups (int, optional): Number of groups for diverse beam search.
            diversity_penalty (float, optional): Diversity penalty for diverse beam search.
            do_sample (bool, optional): Whether to use sampling for generation.
            temperature (float, optional): Sampling temperature.
            top_k (int, optional): Top-k sampling parameter.
            top_p (float, optional): Top-p sampling parameter.

        Returns:
            GenerationOutputs: The generated sequences and their scores.
        """
        outputs = super().generate(
            input_ids=input_ids,
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
