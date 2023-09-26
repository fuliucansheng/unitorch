# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from unitorch.utils.decorators import replace
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)


@replace(transformers.models.llama.modeling_llama.LlamaAttention)
class LlamaAttentionV2(transformers.models.llama.modeling_llama.LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            dtype_min = torch.tensor(
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            attn_weights = torch.max(attn_weights, dtype_min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaForClassification(GenericModel, QuantizationMixin):
    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Llama model for classification tasks.

        Args:
            config_path (str): Path to the model configuration file.
            num_classes (int, optional): Number of classes for classification. Defaults to 1.
            hidden_dropout_prob (float, optional): Dropout probability for hidden layers. Defaults to 0.1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = LlamaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = LlamaModel(self.config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the classification model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output logits.Tensor: tensor of shape (batch_size, num_classes).
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
        pooled_output = outputs[:, -1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class LlamaForPretrain(GenericModel):
    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Llama model for pretraining tasks.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = LlamaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = LlamaModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.init_weights()

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids_label: Optional[torch.Tensor] = None,
        attention_mask_label: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the pretraining model.

        Args:
            input_ids (torch.Tensor, optional): Input tensor of shape (batch_size, sequence_length). Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.
            input_ids_label (torch.Tensor, optional): Labeled input tensor of shape (batch_size, sequence_length). Defaults to None.
            attention_mask_label (torch.Tensor, optional): Labeled attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output loss.Tensor: Loss value.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        predict_logits = self.lm_head(outputs[0])
        batch_size, seq_len, num_classes = predict_logits.size()
        logits = predict_logits.contiguous().view(batch_size * seq_len, num_classes)
        targets = input_ids_label.contiguous().view(-1).long()
        masks = attention_mask_label.contiguous().view(-1)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        loss = loss * masks.float()
        loss = loss.contiguous().view(batch_size, seq_len).sum(1) / torch.max(
            masks.contiguous().view(batch_size, seq_len).float().sum(1),
            torch.ones(batch_size).to(masks.device),
        )
        loss = torch.mean(loss)
        return loss


class LlamaForGeneration(GenericModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        "^(?!model\.model\.|model\.lm_head\.)model\.": "model.",
        "^lm_head.": "model.",
    }

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Llama model for text generation tasks.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = LlamaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = LlamaForCausalLM(self.config)
        self.init_weights()

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = outputs.logits
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
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
        outputs = self.model.generate(
            input_ids,
            max_length=max_gen_seq_length + input_seq_length,
            min_length=min_gen_seq_length + input_seq_length,
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
        outputs.sequences[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )
