# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import math
import random
import logging
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.models.vit.modeling_vit import ViTConfig, ViTModel
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2Config,
    Blip2VisionModel,
    Blip2QFormerModel,
)
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.clip.modeling import AllGather


class MiniGPT4ViTLlamaModel(nn.Module):
    def __init__(
        self,
        blip2_config: Blip2Config,
        llama_config: LlamaConfig,
    ):
        super().__init__()
        self.blip2_config = blip2_config
        self.vision_model = Blip2VisionModel(self.blip2_config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                self.blip2_config.num_query_tokens,
                self.blip2_config.qformer_config.hidden_size,
            )
        )
        self.qformer = Blip2QFormerModel(self.blip2_config.qformer_config)

        self.llama_config = llama_config
        self.language_projection = nn.Linear(
            self.blip2_config.qformer_config.hidden_size, self.llama_config.hidden_size
        )
        self.llama = LlamaForCausalLM(self.llama_config)

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
        vision_outputs = self.vit(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_embeds = query_outputs[0]

        language_model_inputs = self.language_projection(query_embeds)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        prefix_inputs_embeds = self.llama.get_input_embeddings()(prefix_input_ids)
        suffix_inputs_embeds = self.llama.get_input_embeddings()(suffix_input_ids)
        decoder_input_embeds = self.llama.get_input_embeddings()(decoder_input_ids)
        inputs_embeds = torch.cat(
            [
                prefix_inputs_embeds,
                language_model_inputs,
                suffix_inputs_embeds,
                decoder_input_embeds,
            ],
            dim=1,
        )
        expected_device = language_model_attention_mask.device

        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(
                prefix_inputs_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        if suffix_attention_mask is None:
            suffix_attention_mask = torch.ones(
                suffix_inputs_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(
                decoder_input_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        attention_mask = torch.cat(
            [
                prefix_attention_mask,
                language_model_attention_mask,
                suffix_attention_mask,
                decoder_attention_mask.to(expected_device),
            ],
            dim=1,
        )
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        prefix_input_ids: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        vision_outputs = self.vit(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_embeds = query_outputs[0]

        inputs_embeds = self.language_projection(query_embeds)

        if prefix_input_ids is not None:
            prefix_inputs_embeds = self.llama.get_input_embeddings()(prefix_input_ids)
            inputs_embeds = torch.cat([prefix_inputs_embeds, inputs_embeds], dim=1)

        if suffix_input_ids is not None:
            suffix_inputs_embeds = self.llama.get_input_embeddings()(suffix_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, suffix_inputs_embeds], dim=1)

        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            **generate_kwargs,
        )

        return outputs


class MiniGPT4ViTLlamaForGeneration(GenericModel):
    prefix_keys_in_state_dict = {
        "^qformer.": "model.",
        "^vision_model.": "model.",
        "^(?!model\.llama\.)model\.": "model.llama.",
        "^lm_head.": "model.llama.",
    }

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Llama model for text generation tasks.

        Args:
            config_path (str): Path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.blip2_config = Blip2Config.from_json_file(blip2_config_path)
        self.llama_config = LlamaConfig.from_json_file(llama_config_path)
        self.llama_config.gradient_checkpointing = gradient_checkpointing
        self.model = MiniGPT4ViTLlamaModel(self.blip2_config, self.llama_config)
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
        Forward pass of the generation model.

        Args:
            input_ids (torch.Tensor, optional): Input tensor of shape (batch_size, sequence_length). Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output logits.Tensor: tensor of shape (batch_size, sequence_length, vocab_size).
        """
        outputs = self.model(
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
        decoder_end_token_id: Optional[int] = 2,
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
            decoder_start_token_id (int, optional): Start token ID for the decoder. Defaults to 2.
            decoder_end_token_id (int, optional): End token ID for the decoder. Defaults to 2.
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
        outputs = self.model.generate(
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
