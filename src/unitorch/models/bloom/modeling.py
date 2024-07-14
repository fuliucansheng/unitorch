# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import BloomModel, BloomConfig, BloomForCausalLM
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin


class BloomForClassification(GenericModel, PeftWeightLoaderMixin):
    """
    A classification model based on the Bloom architecture.
    """

    prefix_keys_in_state_dict = {"^(?!transformer\.).*": "transformer."}
    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": "transformer."}

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes a new instance of the BloomForClassification model.

        Args:
            config_path (str): The path to the configuration file for the Bloom model.
            num_classes (Optional[int]): The number of output classes for classification. Defaults to 1.
            hidden_dropout_prob (Optional[float]): The dropout probability for the hidden layers. Defaults to 0.1.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = BloomConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.transformer = BloomModel(self.config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the BloomForClassification model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (Optional[torch.Tensor]): The attention mask tensor. Defaults to None.

        Returns:
            (torch.Tensor):The output logits for classification.
        """
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = outputs[:, -1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BloomForGeneration(GenericModel, PeftWeightLoaderMixin):
    """
    A generation model based on the Bloom architecture.
    """

    prefix_keys_in_state_dict = {"^(?!model\.).*": "model.transformer."}
    replace_keys_in_peft_state_dict = {"peft_model.base_model.": ""}

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes a new instance of the BloomForGeneration model.

        Args:
            config_path (str): The path to the configuration file for the Bloom model.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = BloomConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = BloomForCausalLM(self.config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the BloomForGeneration model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (Optional[torch.Tensor]): The attention mask tensor. Defaults to None.

        Returns:
            (torch.Tensor):The output logits.
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
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
    ) -> GenericOutputs:
        """
        Generate sequences using the BloomForGeneration model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            num_beams (Optional[int]): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (Optional[int]): The ID of the start token for decoding. Defaults to 1.
            decoder_end_token_id (Optional[int]): The ID of the end token for decoding. Defaults to 2.
            num_return_sequences (Optional[int]): The number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (Optional[int]): The minimum length of the generated sequences. Defaults to 0.
            max_gen_seq_length (Optional[int]): The maximum length of the generated sequences. Defaults to 48.
            repetition_penalty (Optional[float]): The penalty for repeated n-grams. Defaults to 1.0.
            no_repeat_ngram_size (Optional[int]): The size of n-grams to prevent repetition. Defaults to 0.
            early_stopping (Optional[bool]): Whether to stop generation early based on specified conditions. Defaults to True.
            length_penalty (Optional[float]): The penalty for longer sequences. Defaults to 1.0.
            num_beam_groups (Optional[int]): The number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (Optional[float]): The penalty for diverse beam search. Defaults to 0.0.
            do_sample (Optional[bool]): Whether to use sampling for generation. Defaults to False.
            temperature (Optional[float]): The temperature for sampling. Defaults to 1.0.
            top_k (Optional[int]): The number of top-k tokens to consider for sampling. Defaults to 50.
            top_p (Optional[float]): The cumulative probability for top-p sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: The generated sequences and their scores.
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
