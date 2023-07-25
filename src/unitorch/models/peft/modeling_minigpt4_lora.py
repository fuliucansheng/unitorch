# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from peft import LoraConfig, PeftModelForCausalLM
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.minigpt4.modeling import LlamaConfig, Blip2Config, MiniGPT4Blip2LlamaModel
from unitorch.models.peft import PeftModelForSequenceClassification, GenericPeftModel


class MiniGPT4LoraForGeneration(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^qformer.": "peft_model.base_model.",
        "^query_tokens": "peft_model.base_model.",
        "^vision_model.": "peft_model.base_model.",
        "^model.language_projection.": "peft_model.base_",
        "^(?!model\.language_projection\.)model\.": "peft_model.base_model.llama.",
        "^lm_head.": "peft_model.base_model.llama.",
    }

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        pad_token_id: Optional[int] = 0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.blip2_config = Blip2Config.from_json_file(blip2_config_path)
        self.blip2_config.pad_token_id = pad_token_id
        self.llama_config = LlamaConfig.from_json_file(llama_config_path)
        self.llama_config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForCausalLM(
            MiniGPT4Blip2LlamaModel(self.blip2_config, self.llama_config), self.peft_config
        )
        self.init_weights()

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
        Performs a forward pass of the MiniGPT4LoraForGeneration model.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            decoder_input_ids (torch.Tensor): The input IDs for the decoder tokens.
            prefix_attention_mask (torch.Tensor, optional): The attention mask for the prefix tokens.
            suffix_attention_mask (torch.Tensor, optional): The attention mask for the suffix tokens.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder tokens.

        Returns:
            logits: The output logits.
        """
        outputs = self.peft_model(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            decoder_input_ids=decoder_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = outputs.logits[
            :, -suffix_input_ids.size(1) - decoder_input_ids.size(1) :, :
        ]
        return logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 1,
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
        Generates sequences using the MiniGPT4LoraForGeneration model.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 1.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 2.
            num_return_sequences (int, optional): The number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): The minimum length of the generated sequences. Defaults to 0.
            max_gen_seq_length (int, optional): The maximum length of the generated sequences. Defaults to 48.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): The size of the n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to stop generation early when all beams are finished. Defaults to True.
            length_penalty (float, optional): The length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): The number of groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): The diversity penalty. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
            top_k (int, optional): The value for top-k sampling. Defaults to 50.
            top_p (float, optional): The value for top-p sampling. Defaults to 1.0.

        Returns:
            outputs (GenericOutputs): The generated sequences and their scores.
        """
        outputs = self.peft_model.generate(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            max_length=max_gen_seq_length,
            min_length=min_gen_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            decoder_start_token_id=decoder_start_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
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
        outputs.sequences = torch.zeros(
            sequences.size(0), num_return_sequences, max_gen_seq_length
        ).to(device=sequences.device)
        outputs.sequences[:, :, : sequences.size(-1)].copy_(
            sequences[:, :, : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )
