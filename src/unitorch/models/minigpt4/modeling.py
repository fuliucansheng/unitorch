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

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2Config,
    Blip2VisionModel,
    Blip2QFormerModel,
)
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.clip.modeling import AllGather


class MiniGPT4Blip2LlamaModel(nn.Module):
    """
    MiniGPT4Blip2LlamaModel is a model that combines the Blip2VisionModel, Blip2QFormerModel, and LlamaForCausalLM
    models for generation. It inherits from the nn.Module class.
    """

    def __init__(
        self,
        blip2_config: Blip2Config,
        llama_config: LlamaConfig,
    ):
        """
        Initializes a MiniGPT4Blip2LlamaModel instance.

        Args:
            blip2_config (Blip2Config): The configuration for the Blip2 model.
            llama_config (LlamaConfig): The configuration for the Llama model.
        """
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
        """
        Performs a forward pass of the MiniGPT4Blip2LlamaModel.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            decoder_input_ids (torch.Tensor): The input IDs for the decoder tokens.
            prefix_attention_mask (torch.Tensor, optional): The attention mask for the prefix tokens.
            suffix_attention_mask (torch.Tensor, optional): The attention mask for the suffix tokens.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder tokens.

        Returns:
            outputs: The model outputs.
        """
        vision_outputs = self.vision_model(
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
        """
        Generates sequences using the MiniGPT4Blip2LlamaModel.

        Args:
            pixel_values (torch.FloatTensor): The pixel values.
            prefix_input_ids (torch.Tensor, optional): The input IDs for the prefix tokens. Defaults to None.
            suffix_input_ids (torch.Tensor, optional): The input IDs for the suffix tokens. Defaults to None.
            **generate_kwargs: Additional keyword arguments for sequence generation.

        Returns:
            outputs: The generation outputs.
        """
        vision_outputs = self.vision_model(
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

        attention_mask = torch.ones(
            inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.bool
        ).to(inputs_embeds.device)

        if prefix_input_ids is not None:
            prefix_inputs_embeds = self.llama.get_input_embeddings()(prefix_input_ids)
            inputs_embeds = torch.cat([prefix_inputs_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [prefix_input_ids.ne(self.blip2_config.pad_token_id), attention_mask],
                dim=1,
            )

        if suffix_input_ids is not None:
            suffix_inputs_embeds = self.llama.get_input_embeddings()(suffix_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, suffix_inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [attention_mask, suffix_input_ids.ne(self.blip2_config.pad_token_id)],
                dim=1,
            )

        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs


class MiniGPT4Blip2LlamaForGeneration(GenericModel):
    """
    MiniGPT4Blip2LlamaForGeneration is a generation model that combines the MiniGPT4Blip2LlamaModel with generation
    capabilities. It inherits from the GenericModel class.
    """

    prefix_keys_in_state_dict = {
        "^qformer.": "model.",
        "^query_tokens": "model.",
        "^vision_model.": "model.",
        "^(?!model\.llama\.|model\.language_projection\.|model\.qformer\.|model\.query_tokens|model\.vision_model\.)model\.": "model.llama.",
        "^lm_head.": "model.llama.",
    }

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        pad_token_id: Optional[int] = 0,
        freeze_vision_model: Optional[bool] = True,
        freeze_qformer_model: Optional[bool] = True,
        freeze_llama_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes a MiniGPT4Blip2LlamaForGeneration instance.

        Args:
            blip2_config_path (str): The path to the Blip2 model configuration file.
            llama_config_path (str): The path to the Llama model configuration file.
            pad_token_id (int, optional): The ID of the padding token. Defaults to 0.
            freeze_vision_model (bool, optional): Whether to freeze the parameters of the vision model. Defaults to True.
            freeze_qformer_model (bool, optional): Whether to freeze the parameters of the qformer model. Defaults to True.
            freeze_llama_model (bool, optional): Whether to freeze the parameters of the llama model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.blip2_config = Blip2Config.from_json_file(blip2_config_path)
        self.blip2_config.pad_token_id = pad_token_id
        self.llama_config = LlamaConfig.from_json_file(llama_config_path)
        self.llama_config.gradient_checkpointing = gradient_checkpointing
        self.model = MiniGPT4Blip2LlamaModel(self.blip2_config, self.llama_config)
        self.init_weights()

        if freeze_vision_model:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False

        if freeze_qformer_model:
            for param in self.model.qformer.parameters():
                param.requires_grad = False

        if freeze_llama_model:
            for param in self.model.llama.parameters():
                param.requires_grad = False

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
        Performs a forward pass of the MiniGPT4Blip2LlamaForGeneration model.

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
        Generates sequences using the MiniGPT4Blip2LlamaForGeneration model.

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
