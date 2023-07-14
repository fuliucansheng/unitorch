# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import MT5Config, MT5Model, MT5ForConditionalGeneration
from transformers.models.mt5.modeling_mt5 import (
    MT5Stack,
    MT5DenseGatedActDense,
    MT5LayerNorm,
    MT5DenseActDense,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs


@replace(transformers.models.mt5.modeling_mt5.MT5Attention)
class MT5AttentionV2(transformers.models.mt5.modeling_mt5.MT5Attention):
    def __init__(
        self,
        config: MT5Config,
        has_relative_attention_bias=False,
    ):
        super().__init__(
            config=config,
            has_relative_attention_bias=has_relative_attention_bias,
        )

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                states.size()[0], -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        kv_batch_size = key_states.size(0)

        # compute scores
        # scores = torch.matmul(
        #     query_states, key_states.transpose(3, 2)
        # )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.einsum(
            "bxnqd,bnkd->bxnqk",
            query_states.view(kv_batch_size, -1, *query_states.size()[1:]),
            key_states,
        )
        scores = scores.reshape(-1, *scores.size()[2:])

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = torch.einsum(
            "bxnqk,bnkd->bxnqd",
            attn_weights.view(kv_batch_size, -1, *attn_weights.size()[1:]),
            value_states,
        )
        attn_output = attn_output.reshape(-1, *attn_output.size()[2:])
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


@replace(transformers.models.mt5.modeling_mt5.MT5LayerFF)
class MT5LayerFF(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = MT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = MT5DenseActDense(config)

        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        with torch.cuda.amp.autocast(enabled=False):
            forwarded_states = self.layer_norm(hidden_states)
            forwarded_states = self.DenseReluDense(forwarded_states)
            hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class MT5ForGeneration(GenericModel):
    prefix_keys_in_state_dict = {"^(?!model\.).*": "model."}

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes an MT5ForGeneration model with the provided configuration.

        Args:
            config_path (str): The path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = MT5Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = MT5ForConditionalGeneration(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        """
        Performs forward pass of the MT5ForGeneration model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of attention mask.
            decoder_input_ids (torch.Tensor): Tensor of decoder input token IDs.
            decoder_attention_mask (torch.Tensor): Tensor of decoder attention mask.

        Returns:
            (torch.Tensor):The model's logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        return logits

    @torch.no_grad()
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
        """
        Generates sequences using the MT5ForGeneration model.

        Args:
            input_ids: The input token IDs.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The decoder's start token ID. Defaults to 2.
            decoder_end_token_id (int or List[int], optional): The decoder's end token ID. Defaults to 2.
            num_return_sequences (int, optional): The number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): The minimum length of the generated sequences. Defaults to 0.
            max_gen_seq_length (int, optional): The maximum length of the generated sequences. Defaults to 48.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): The size of n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to stop generation early. Defaults to True.
            length_penalty (float, optional): The length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): The number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): The diversity penalty. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): The temperature for sampling. Defaults to 1.0.
            top_k (int, optional): The value for top-k sampling. Defaults to 50.
            top_p (float, optional): The value for top-p (nucleus) sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: The generated sequences and their scores.
        """
        outputs = self.model.generate(
            input_ids,
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
        outputs.sequences[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )
