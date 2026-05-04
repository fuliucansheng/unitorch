# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import List, Optional, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.pegasus import PegasusForGeneration as _PegasusForGeneration
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import GenerationOutputs
from unitorch.cli.models.pegasus import pretrained_pegasus_infos


@register_model("core/model/generation/pegasus", generation_model_decorator)
class PegasusForGeneration(_PegasusForGeneration):
    """Pegasus model for generation tasks."""

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path, gradient_checkpointing=gradient_checkpointing
        )

    @classmethod
    @config_defaults_init("core/model/generation/pegasus")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/generation/pegasus")
        pretrained_name = config.getoption("pretrained_name", "pegasus-xsum")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_pegasus_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_pegasus_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @config_defaults_method("core/model/generation/pegasus")
    @torch.no_grad()
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 0,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 1,
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
