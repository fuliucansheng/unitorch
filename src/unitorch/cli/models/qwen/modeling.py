# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import List, Optional, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value, is_bfloat16_available
from unitorch.models.qwen import (
    QWen3ForGeneration as _QWen3ForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import GenerationOutputs
from unitorch.cli.models.qwen import (
    pretrained_qwen_infos,
    pretrained_qwen_extensions_infos,
)


@register_model("core/model/generation/qwen3", generation_model_decorator)
class QWen3ForGeneration(_QWen3ForGeneration):
    """Qwen3 model for text generation."""

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
    @add_default_section_for_init("core/model/generation/qwen3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/generation/qwen3")
        pretrained_name = config.getoption("pretrained_name", "qwen3-4b-thinking")
        pretrained_lora_name = config.getoption("pretrained_lora_name", None)
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = pop_value(
            pretrained_lora_weight_path,
            nested_dict_value(pretrained_qwen_extensions_infos, pretrained_lora_name),
            check_none=False,
        )
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
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/qwen3")
    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 151643,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 151645,
        decoder_pad_token_id: Optional[int] = 151643,
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
            input_ids,
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
