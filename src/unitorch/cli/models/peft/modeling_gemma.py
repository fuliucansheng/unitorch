# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

import torch
from torch import autocast

from unitorch.utils import is_bfloat16_available
from unitorch.models.peft import GemmaLoraForGeneration as _GemmaLoraForGeneration
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_model,
)
from unitorch.cli.models import GenerationOutputs, generation_model_decorator
from unitorch.cli.models.gemma import resolve_pretrained_gemma_path


@register_model("core/model/generation/peft/lora/gemma", generation_model_decorator)
class GemmaLoraForGeneration(_GemmaLoraForGeneration):
    """Gemma LoRA model for text generation."""

    def __init__(
        self,
        config_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = False,
        target_modules: Optional[Union[List[str], str]] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @config_defaults_init("core/model/generation/peft/lora/gemma")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/generation/peft/lora/gemma")
        pretrained_name = config.getoption("pretrained_name", "gemma-4-12b")

        config_path = config.getoption("config_path", None)
        if config_path is None:
            config_path = resolve_pretrained_gemma_path(pretrained_name, "config")
        else:
            config_path = cached_path(config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", False)
        target_modules = config.getoption(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = (
            pretrained_weight_path
            if pretrained_weight_path is not None
            else resolve_pretrained_gemma_path(pretrained_name, "weight")
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(cached_path(pretrained_weight_path))
            else:
                weight_path.extend(map(cached_path, pretrained_weight_path))

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            weight_path.append(cached_path(pretrained_lora_weight_path))

        if len(weight_path) > 0:
            inst.from_pretrained(weight_path=weight_path)

        return inst

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @config_defaults_method("core/model/generation/peft/lora/gemma")
    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 2,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 1,
        decoder_pad_token_id: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 512,
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            decoder_pad_token_id=decoder_pad_token_id,
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
