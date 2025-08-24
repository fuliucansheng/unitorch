# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from peft import LoraConfig, PeftModelForCausalLM
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
)
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftModelForSequenceClassification, GenericPeftModel


class QWen2_5VLLoraForGeneration(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^visual.": "model.model.",
        "^model(?!\.model).": "model.model.language_",
    }
    replace_keys_in_state_dict = {
        "q_proj.weight": "q_proj.base_layer.weight",
        "q_proj.bias": "q_proj.base_layer.bias",
        "v_proj.weight": "v_proj.base_layer.weight",
        "v_proj.bias": "v_proj.base_layer.bias",
    }

    def __init__(
        self,
        config_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Bloom Lora model for text generation tasks.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = Qwen2_5_VLConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.model = Qwen2_5_VLForConditionalGeneration(self.config)
        self.model.add_adapter(self.peft_config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the generation model.

        Args:
            input_ids (torch.Tensor, optional): Input tensor of shape (batch_size, sequence_length). Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output logits.Tensor: tensor of shape (batch_size, sequence_length, vocab_size).
        """
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
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
        """
        Generate text using the generation model.

        Args:
            input_ids: Input tensor of shape (batch_size, sequence_length).
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 2.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 2.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum length of generated sequences. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum length of generated sequences. Defaults to 48.
            repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to stop generation early. Defaults to True.
            length_penalty (float, optional): Penalty for longer sequences. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Penalty for diverse sequences in diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k value for sampling. Defaults to 50.
            top_p (float, optional): Top-p value for sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: Generated sequences and their scores.
        """
        input_seq_length = input_ids.size(1)
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))
        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
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
        outputs.sequences = (
            torch.zeros(sequences.size(0), num_return_sequences, max_gen_seq_length).to(
                device=sequences.device
            )
            + decoder_start_token_id
        )
        outputs.sequences[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )


class QWen2_5VLDPOLoraForGeneration(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^visual.": "model.model.",
        "^model(?!\.model).": "model.model.language_",
    }
    replace_keys_in_state_dict = {
        "q_proj.weight": "q_proj.base_layer.weight",
        "q_proj.bias": "q_proj.base_layer.bias",
        "v_proj.weight": "v_proj.base_layer.weight",
        "v_proj.bias": "v_proj.base_layer.bias",
    }

    def __init__(
        self,
        config_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Bloom Lora model for text generation tasks.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = Qwen2_5_VLConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.model = Qwen2_5_VLForConditionalGeneration(self.config)
        self.model.add_adapter(self.peft_config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        win_input_ids: torch.Tensor,
        lose_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        win_attention_mask: Optional[torch.Tensor] = None,
        lose_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the generation model.

        Args:
            input_ids (torch.Tensor, optional): Input tensor of shape (batch_size, sequence_length). Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output logits.Tensor: tensor of shape (batch_size, sequence_length, vocab_size).
        """
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))
        win_input_ids = torch.cat([input_ids, win_input_ids], dim=1)
        lose_input_ids = torch.cat([input_ids, lose_input_ids], dim=1)
        if attention_mask is not None and win_attention_mask is not None:
            win_attention_mask = torch.cat([attention_mask, win_attention_mask], dim=1)
        if attention_mask is not None and lose_attention_mask is not None:
            lose_attention_mask = torch.cat(
                [attention_mask, lose_attention_mask], dim=1
            )
        win_outputs = self.model(
            input_ids=win_input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=win_attention_mask,
            return_dict=True,
        )
        lose_outputs = self.model(
            input_ids=lose_input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=lose_attention_mask,
            return_dict=True,
        )
        input_seq_length = input_ids.size(1)
        win_logits = win_outputs.logits[:, input_seq_length - 1 : -1, :]
        lose_logits = lose_outputs.logits[:, input_seq_length - 1 : -1, :]
        win_labels = win_input_ids[:, input_seq_length:]
        lose_labels = lose_input_ids[:, input_seq_length:]
        win_labels_mask = win_attention_mask[:, input_seq_length:]
        lose_labels_mask = lose_attention_mask[:, input_seq_length:]
        win_nll_loss = F.cross_entropy(
            win_logits.reshape(-1, win_logits.size(-1)),
            win_labels.reshape(-1),
            reduction="none",
        ).reshape(win_labels.size(0), -1)
        win_logprobs = -win_nll_loss * win_labels_mask
        lose_nll_loss = F.cross_entropy(
            lose_logits.reshape(-1, lose_logits.size(-1)),
            lose_labels.reshape(-1),
            reduction="none",
        ).reshape(lose_labels.size(0), -1)
        lose_logprobs = -lose_nll_loss * lose_labels_mask

        with torch.no_grad():
            self.model.disable_adapters()
            ref_win_outputs = self.model(
                input_ids=win_input_ids,
                attention_mask=win_attention_mask,
                return_dict=True,
            )
            ref_lose_outputs = self.model(
                input_ids=lose_input_ids,
                attention_mask=lose_attention_mask,
                return_dict=True,
            )
            ref_win_logits = ref_win_outputs.logits[:, input_seq_length - 1 : -1, :]
            ref_lose_logits = ref_lose_outputs.logits[:, input_seq_length - 1 : -1, :]
            ref_win_nll_loss = F.cross_entropy(
                ref_win_logits.reshape(-1, ref_win_logits.size(-1)),
                win_labels.reshape(-1),
                reduction="none",
            ).reshape(win_labels.size(0), -1)
            ref_win_logprobs = -ref_win_nll_loss * win_labels_mask
            ref_lose_nll_loss = F.cross_entropy(
                ref_lose_logits.reshape(-1, ref_lose_logits.size(-1)),
                lose_labels.reshape(-1),
                reduction="none",
            ).reshape(lose_labels.size(0), -1)
            ref_lose_logprobs = -ref_lose_nll_loss * lose_labels_mask
            self.model.enable_adapters()

        logratios = win_logprobs - lose_logprobs
        ref_logratios = ref_win_logprobs - ref_lose_logprobs
        logits = logratios - ref_logratios
        loss = -F.logsigmoid(self.dpo_beta * logits).mean()
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
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
        """
        Generate text using the generation model.

        Args:
            input_ids: Input tensor of shape (batch_size, sequence_length).
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 2.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 2.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum length of generated sequences. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum length of generated sequences. Defaults to 48.
            repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to stop generation early. Defaults to True.
            length_penalty (float, optional): Penalty for longer sequences. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Penalty for diverse sequences in diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k value for sampling. Defaults to 50.
            top_p (float, optional): Top-p value for sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: Generated sequences and their scores.
        """
        input_seq_length = input_ids.size(1)
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))
        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
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
        outputs.sequences = (
            torch.zeros(sequences.size(0), num_return_sequences, max_gen_seq_length).to(
                device=sequences.device
            )
            + decoder_start_token_id
        )
        outputs.sequences[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )
