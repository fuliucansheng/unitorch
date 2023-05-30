# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import PegasusXConfig, PegasusModel, PegasusXForConditionalGeneration
from transformers.models.pegasus_x.modeling_pegasus_x import (
    PegasusXEncoder,
    PegasusXDecoder,
)

from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs


@replace(transformers.models.pegasus_x.modeling_pegasus_x.PegasusXAttention)
class PegasusXAttentionV2(
    transformers.models.pegasus_x.modeling_pegasus_x.PegasusXAttention
):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: Optional[float] = 0.0,
        is_decoder: Optional[bool] = False,
        bias: Optional[bool] = True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        kv_bsz = bsz

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
            kv_bsz = key_states.size(0)
        elif is_cross_attention:
            # cross_attentions
            kv_bsz = key_value_states.size(0)
            key_states = self._shape(self.k_proj(key_value_states), -1, kv_bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, kv_bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        kv_proj_shape = (kv_bsz * self.num_heads, -1, self.head_dim)
        key_states = key_states.view(*kv_proj_shape)
        value_states = value_states.view(*kv_proj_shape)

        src_len = key_states.size(1)

        if is_cross_attention and kv_bsz != bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                query_states.view(kv_bsz, -1, self.num_heads, *query_states.size()[1:]),
                key_states.view(kv_bsz, self.num_heads, *key_states.size()[1:]),
            )
            attn_weights = attn_weights.reshape(-1, *attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        if is_cross_attention and bsz != kv_bsz:
            attn_probs = attn_probs.to(value_states.dtype)
            attn_output = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(kv_bsz, -1, self.num_heads, *attn_probs.size()[1:]),
                value_states.view(kv_bsz, self.num_heads, *value_states.size()[1:]),
            )
            attn_output = attn_output.reshape(-1, *attn_output.size()[-2:])
        else:
            attn_probs = attn_probs.to(value_states.dtype)
            attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class XPegasusForGeneration(GenericModel):
    """
    XPegasus model for text generation tasks.
    """

    prefix_keys_in_state_dict = {
        "^(?!model\.model\.)model\.": "model.",
        "^lm_head.": "model.",
    }

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the XPegasusForGeneration model.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = PegasusXConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = PegasusXForConditionalGeneration(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        """
        Performs forward pass of the XPegasusForGeneration model.

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
        decoder_end_token_id: Optional[int] = 1,
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
        Generates sequences using the XPegasusForGeneration model.

        Args:
            input_ids: The input token IDs.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The decoder's start token ID. Defaults to 2.
            decoder_end_token_id (int, optional): The decoder's end token ID. Defaults to 2.
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
