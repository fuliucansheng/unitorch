# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

import torch
from torch import autocast

from unitorch.utils import is_bfloat16_available
from unitorch.models.gemma import GemmaVLForGeneration as _GemmaVLForGeneration
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_model,
)
from unitorch.cli.models import GenerationOutputs, generation_model_decorator
from unitorch.cli.models.gemma import (
    pretrained_gemma_extensions_infos,
    resolve_pretrained_gemma_path,
)


@register_model("core/model/generation/gemma_vl", generation_model_decorator)
class GemmaVLForGeneration(_GemmaVLForGeneration):
    """Gemma multimodal model for image-grounded generation."""

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @config_defaults_init("core/model/generation/gemma_vl")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/generation/gemma_vl")
        pretrained_name = config.getoption("pretrained_name", "gemma-4-12b")
        pretrained_lora_name = config.getoption("pretrained_lora_name", None)

        config_path = config.getoption("config_path", None)
        if config_path is None:
            config_path = resolve_pretrained_gemma_path(pretrained_name, "config")
        else:
            config_path = cached_path(config_path)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        inst = cls(
            config_path=config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = (
            pretrained_weight_path
            if pretrained_weight_path is not None
            else resolve_pretrained_gemma_path(pretrained_name, "weight")
        )
        if pretrained_weight_path is not None:
            weight_path = cached_path(weight_path)
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = (
            pretrained_lora_weight_path
            if pretrained_lora_weight_path is not None
            else pretrained_gemma_extensions_infos.get(pretrained_lora_name)
        )
        if pretrained_lora_weight_path is not None:
            lora_weight_path = cached_path(lora_weight_path)
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if lora_weight_path is not None:
            inst.load_lora_weights(
                lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        return GenerationOutputs(sequences=outputs)

    @config_defaults_method("core/model/generation/gemma_vl")
    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
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
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
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
