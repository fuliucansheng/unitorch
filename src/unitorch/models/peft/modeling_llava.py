# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import math
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from peft import LoraConfig, PeftModelForCausalLM
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextConfig,
    LlavaNextMultiModalProjector,
)
from transformers import (
    MistralModel,
    MistralConfig,
    MistralForCausalLM,
    CLIPVisionModel,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.quantization import quantize_model
from unitorch.models.peft import PeftModelForSequenceClassification, GenericPeftModel


class LlavaMistralClipLoraForClassification(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^language_model.": "peft_model.base_model.",
    }
    replace_keys_in_state_dict = {
        r"(language_model.*?)q_proj\.weight": r"\1q_proj.base_layer.weight",
        r"(language_model.*?)v_proj\.weight": r"\1v_proj.base_layer.weight",
        "language_model.": "",
    }
    modules_to_save_checkpoints = ["lora", "multi_modal_projector", "classifier"]

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        image_token_index: Optional[int] = 32000,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
        freeze_multi_modal_projector: Optional[bool] = True,
        freeze_classifer: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = LlavaNextConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.vision_tower = CLIPVisionModel(self.config.vision_config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(self.config)
        embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)
        self.image_newline = nn.Parameter(
            torch.randn(self.config.text_config.hidden_size, dtype=self.dtype)
            * embed_std
        )
        language_model = MistralModel(self.config.text_config)

        if quant_config_path is not None:
            quant_config = QuantizationConfig.from_json_file(quant_config_path)
            ignore_modules = target_modules + ["lm_head"]
            language_model = quantize_model(
                language_model, quant_config, ignore_modules=ignore_modules
            )
        self.peft_model = PeftModelForSequenceClassification(
            language_model, self.peft_config
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.text_config.hidden_size, num_classes)
        self.init_weights()

        if freeze_classifer:
            for param in self.classifier.parameters():
                param.requires_grad = False

        for param in self.vision_tower.parameters():
            param.requires_grad = False

        if freeze_multi_modal_projector:
            for param in self.multi_modal_projector.parameters():
                param.requires_grad = False

        self.image_token_index = image_token_index

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        image_embeds = vision_outputs.hidden_states[-2][:, 1:]
        image_embeds = self.multi_modal_projector(image_embeds)
        image_embeds = torch.cat(
            [
                image_embeds,
                self.image_newline.expand(
                    image_embeds.shape[0], 1, image_embeds.shape[-1]
                ),
            ],
            dim=1,
        )

        image_seq_length = image_embeds.size(1)
        batch_indices, text_indices = torch.where(input_ids != self.image_token_index)
        image_masks = (input_ids == self.image_token_index).long() * (
            image_seq_length - 1
        )
        new_positions = torch.cumsum(image_masks + 1, dim=1) - 1
        new_text_indices = new_positions[batch_indices, text_indices]

        input_ids[input_ids == self.image_token_index] = 0
        text_embeds = self.peft_model.base_model.get_input_embeddings()(input_ids)

        batch_size, text_seq_length, text_dim = text_embeds.size()

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, text_seq_length).to(
                text_embeds.device
            )

        final_embeds = torch.zeros(
            batch_size, text_seq_length + image_seq_length - 1, text_dim
        ).to(text_embeds.device)
        overwrite_masks = torch.ones(
            batch_size, text_seq_length + image_seq_length - 1
        ).to(text_embeds.device)
        overwrite_masks[batch_indices, new_text_indices] = 0
        final_embeds[overwrite_masks == 0] = text_embeds[
            batch_indices, text_indices
        ].to(final_embeds)
        final_embeds[overwrite_masks == 1] = (
            image_embeds.contiguous().view(-1, text_dim).to(final_embeds)
        )
        final_masks = torch.zeros(
            batch_size, text_seq_length + image_seq_length - 1
        ).to(attention_mask)
        final_masks[overwrite_masks == 0] = attention_mask[
            batch_indices, text_indices
        ].to(final_masks)
        final_masks[overwrite_masks == 1] = 1
        position_ids = (final_masks.cumsum(dim=1) - 1).masked_fill(final_masks == 0, -1)

        outputs = self.peft_model(
            inputs_embeds=final_embeds,
            attention_mask=final_masks,
            position_ids=position_ids,
        )[0]
        pooled_output = outputs[:, -1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class LlavaMistralClipLoraForGeneration(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^language_model.": "peft_model.base_model.",
    }
    replace_keys_in_state_dict = {
        r"(language_model.*?)q_proj\.weight": r"\1q_proj.base_layer.weight",
        r"(language_model.*?)v_proj\.weight": r"\1v_proj.base_layer.weight",
        "language_model\.": "model.",
    }

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        image_token_index: Optional[int] = 32000,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        freeze_multi_modal_projector: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        pad_token_id: Optional[int] = 0,
    ):
        super().__init__()
        self.config = LlavaNextConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.pad_token_id = pad_token_id
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.vision_tower = CLIPVisionModel(self.config.vision_config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(self.config)
        embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)
        self.image_newline = nn.Parameter(
            torch.randn(self.config.text_config.hidden_size, dtype=self.dtype)
            * embed_std
        )
        language_model = MistralForCausalLM(self.config.text_config)

        if quant_config_path is not None:
            quant_config = QuantizationConfig.from_json_file(quant_config_path)
            ignore_modules = target_modules + ["lm_head"]
            language_model = quantize_model(
                language_model, quant_config, ignore_modules=ignore_modules
            )

        self.peft_model = PeftModelForCausalLM(language_model, self.peft_config)
        self.init_weights()

        for param in self.vision_tower.parameters():
            param.requires_grad = False

        if freeze_multi_modal_projector:
            for param in self.multi_modal_projector.parameters():
                param.requires_grad = False

        self.image_token_index = image_token_index

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
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
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        image_embeds = vision_outputs.hidden_states[-2][:, 1:]
        image_embeds = self.multi_modal_projector(image_embeds)
        image_embeds = torch.cat(
            [
                image_embeds,
                self.image_newline.expand(
                    image_embeds.shape[0], 1, image_embeds.shape[-1]
                ),
            ],
            dim=1,
        )

        image_seq_length = image_embeds.size(1)
        batch_indices, text_indices = torch.where(input_ids != self.image_token_index)
        image_masks = (input_ids == self.image_token_index).long() * (
            image_seq_length - 1
        )
        new_positions = torch.cumsum(image_masks + 1, dim=1) - 1
        new_text_indices = new_positions[batch_indices, text_indices]

        input_ids[input_ids == self.image_token_index] = 0
        text_embeds = self.peft_model.base_model.get_input_embeddings()(input_ids)

        batch_size, text_seq_length, text_dim = text_embeds.size()

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, text_seq_length).to(
                text_embeds.device
            )

        final_embeds = torch.zeros(
            batch_size, text_seq_length + image_seq_length - 1, text_dim
        ).to(text_embeds.device)
        overwrite_masks = torch.ones(
            batch_size, text_seq_length + image_seq_length - 1
        ).to(text_embeds.device)
        overwrite_masks[batch_indices, new_text_indices] = 0
        final_embeds[overwrite_masks == 0] = text_embeds[
            batch_indices, text_indices
        ].to(final_embeds)
        final_embeds[overwrite_masks == 1] = (
            image_embeds.contiguous().view(-1, text_dim).to(final_embeds)
        )
        final_masks = torch.zeros(
            batch_size, text_seq_length + image_seq_length - 1
        ).to(attention_mask)
        final_masks[overwrite_masks == 0] = attention_mask[
            batch_indices, text_indices
        ].to(final_masks)
        final_masks[overwrite_masks == 1] = 1
        position_ids = (final_masks.cumsum(dim=1) - 1).masked_fill(final_masks == 0, -1)

        outputs = self.peft_model(
            inputs_embeds=final_embeds,
            attention_mask=final_masks,
            position_ids=position_ids,
        )
        logits = torch.zeros(batch_size, text_seq_length, outputs.logits.size(-1)).to(
            outputs.logits.device
        )
        logits[batch_indices, text_indices] = outputs.logits[
            batch_indices, new_text_indices
        ]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        image_embeds = vision_outputs.hidden_states[-2][:, 1:]
        image_embeds = self.multi_modal_projector(image_embeds)
        image_embeds = torch.cat(
            [
                image_embeds,
                self.image_newline.expand(
                    image_embeds.shape[0], 1, image_embeds.shape[-1]
                ),
            ],
            dim=1,
        )

        image_seq_length = image_embeds.size(1)
        batch_indices, text_indices = torch.where(input_ids != self.image_token_index)
        image_masks = (input_ids == self.image_token_index).long() * (
            image_seq_length - 1
        )
        new_positions = torch.cumsum(image_masks + 1, dim=1) - 1
        new_text_indices = new_positions[batch_indices, text_indices]

        input_ids[input_ids == self.image_token_index] = 0
        text_embeds = self.peft_model.base_model.get_input_embeddings()(input_ids)

        batch_size = text_embeds.size(0)
        text_seq_length, image_seq_length = text_embeds.size(1), image_embeds.size(1)
        text_dim = text_embeds.size(2)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, text_seq_length).to(
                text_embeds.device
            )

        final_embeds = torch.zeros(
            batch_size, text_seq_length + image_seq_length - 1, text_dim
        ).to(text_embeds.device)
        overwrite_masks = torch.ones(
            batch_size, text_seq_length + image_seq_length - 1
        ).to(text_embeds.device)
        overwrite_masks[batch_indices, new_text_indices] = 0
        final_embeds[overwrite_masks == 0] = text_embeds[
            batch_indices, text_indices
        ].to(final_embeds)
        final_embeds[overwrite_masks == 1] = (
            image_embeds.contiguous().view(-1, text_dim).to(final_embeds)
        )
        final_masks = torch.zeros(
            batch_size, text_seq_length + image_seq_length - 1
        ).to(attention_mask)
        final_masks[overwrite_masks == 0] = attention_mask[
            batch_indices, text_indices
        ].to(final_masks)
        final_masks[overwrite_masks == 1] = 1
        input_seq_length = final_embeds.size(1)
        outputs = self.peft_model.generate(
            inputs_embeds=final_embeds,
            attention_mask=final_masks,
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
