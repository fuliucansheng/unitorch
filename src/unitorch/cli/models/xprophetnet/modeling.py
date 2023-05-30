# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.xprophetnet import (
    XProphetNetForGeneration as _XProphetNetForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import GenerationOutputs
from unitorch.cli.models.xprophetnet import pretrained_xprophetnet_infos


@register_model("core/model/generation/xprophetnet", generation_model_decorator)
class XProphetNetForGeneration(_XProphetNetForGeneration):
    """XProphetNet model for generation tasks."""
    def __init__(
        self,
        config_path: str,
        freeze_input_embedding: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize XProphetNetForGeneration.

        Args:
            config_path (str): The path to the model configuration file.
            freeze_input_embedding (bool, optional): Whether to freeze the input embeddings. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            freeze_input_embedding=freeze_input_embedding,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/xprophetnet")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of XProphetNetForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            XProphetNetForGeneration: An instance of XProphetNetForGeneration.
        """
        config.set_default_section("core/model/generation/xprophetnet")
        pretrained_name = config.getoption("pretrained_name", "default-xprophetnet")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_xprophetnet_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        freeze_input_embedding = config.getoption("freeze_input_embedding", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            freeze_input_embedding=freeze_input_embedding,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_xprophetnet_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)

        return inst

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        """
        Forward pass of the XProphetNetForGeneration model.

        Args:
            input_ids (torch.Tensor): Input tensor IDs.
            attention_mask (torch.Tensor): Attention mask tensor.
            decoder_input_ids (torch.Tensor): Decoder input tensor IDs.
            decoder_attention_mask (torch.Tensor): Decoder attention mask tensor.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/xprophetnet")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 2,
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
        Generate sequences using the XProphetNetForGeneration model.

        Args:
            input_ids (torch.Tensor): Input tensor IDs.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 2.
            decoder_end_token_id (int, optional): The ID of the decoder end token. Defaults to 2.
            num_return_sequences (int, optional): The number of returned sequences. Defaults to 1.
            min_gen_seq_length (int, optional): The minimum generated sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to 48.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): The size of n-grams that should not be repeated. Defaults to 0.
            early_stopping (bool, optional): Whether to stop generation early. Defaults to True.
            length_penalty (float, optional): The length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): The number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): The diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): The temperature for sampling. Defaults to 1.0.
            top_k (int, optional): The top-k value for sampling. Defaults to 50.
            top_p (float, optional): The top-p value for sampling. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().generate(
            input_ids,
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
