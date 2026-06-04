# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

import torch
from peft import LoraConfig
from transformers.models.gemma4 import Gemma4ForConditionalGeneration

from unitorch.models import GenericOutputs
from unitorch.models.gemma.modeling import _get_gemma_dtype, _get_gemma_text_config
from unitorch.models.peft import GenericPeftModel, add_adapter_compat


class GemmaLoraForGeneration(GenericPeftModel):
    prefix_keys_in_state_dict = {"^(?!model\\.model\\.).*": "model."}
    replace_keys_in_state_dict = {
        "q_proj.weight": "q_proj.base_layer.weight",
        "q_proj.bias": "q_proj.base_layer.bias",
        "k_proj.weight": "k_proj.base_layer.weight",
        "k_proj.bias": "k_proj.base_layer.bias",
        "v_proj.weight": "v_proj.base_layer.weight",
        "v_proj.bias": "v_proj.base_layer.bias",
        "o_proj.weight": "o_proj.base_layer.weight",
        "o_proj.bias": "o_proj.base_layer.bias",
    }

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
        super().__init__()
        self.config = _get_gemma_text_config(config_path)
        if gradient_checkpointing:
            self.config.use_cache = False
            if getattr(self.config, "text_config", None) is not None:
                self.config.text_config.use_cache = False
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.model = Gemma4ForConditionalGeneration(self.config)
        self.model = add_adapter_compat(self.model, self.peft_config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.init_weights()
        self.model.to(dtype=_get_gemma_dtype(self.config))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    @torch.no_grad()
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
        input_seq_length = input_ids.size(1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_seq_length + input_seq_length,
            min_length=min_gen_seq_length + input_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
            pad_token_id=decoder_pad_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences.reshape(
            -1, num_return_sequences, outputs.sequences.size(-1)
        )
        padded = torch.full(
            (sequences.size(0), num_return_sequences, max_gen_seq_length),
            fill_value=decoder_pad_token_id,
            device=sequences.device,
        )
        padded[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            padded = padded.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=padded.long(),
            sequences_scores=getattr(outputs, "sequences_scores", None),
        )
