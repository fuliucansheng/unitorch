# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.blip.modeling_blip import (
    BlipConfig,
    BlipForConditionalGeneration,
    BlipTextModel,
    BlipVisionModel,
)

from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.clip.modeling import AllGather


def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def _blip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return (_contrastive_loss(similarity) + _contrastive_loss(similarity.T)) / 2.0


def _freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


class BlipForPretrain(GenericModel):
    """BLIP model for contrastive image-text pre-training."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
        use_all_gather: bool = True,
    ) -> None:
        super().__init__()
        config = BlipConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.use_all_gather = use_all_gather

        self.text_model = BlipTextModel(config.text_config)
        self.vision_model = BlipVisionModel(config.vision_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, projection_dim, bias=False)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)
        self.init_weights()

        if freeze_base_model:
            _freeze(self.text_model)
            _freeze(self.vision_model)

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        out = AllGather.apply(x)
        return out.view(-1, *out.shape[2:])

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.visual_projection(
            self.vision_model(pixel_values=pixel_values).pooler_output
        )
        text_embeds = self.text_projection(
            self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).pooler_output
        )
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        if self.use_all_gather and dist.is_initialized():
            text_embeds = self._all_gather(text_embeds)
            image_embeds = self._all_gather(image_embeds)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
        return _blip_loss(logits_per_text)


class BlipForClassification(GenericModel):
    """BLIP model for multimodal (image + text) classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_classes: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = BlipConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.text_model = BlipTextModel(config.text_config)
        self.vision_model = BlipVisionModel(config.vision_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, projection_dim, bias=False)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, projection_dim, bias=False)
        self.classifier = nn.Linear(projection_dim * 2, num_classes)
        self.init_weights()

        if freeze_base_model:
            _freeze(self.text_model)
            _freeze(self.vision_model)

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.visual_projection(
            self.vision_model(pixel_values=pixel_values).pooler_output
        )
        text_embeds = self.text_projection(
            self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).pooler_output
        )
        return self.classifier(F.relu(torch.cat([image_embeds, text_embeds], dim=1)))


class BlipForTextClassification(GenericModel):
    """BLIP model for text-only classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_classes: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = BlipConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing

        self.text_model = BlipTextModel(config.text_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, projection_dim, bias=False)
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.init_weights()

        if freeze_base_model:
            _freeze(self.text_model)

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_embeds = self.text_projection(
            self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).pooler_output
        )
        return self.classifier(F.relu(text_embeds))


class BlipForImageClassification(GenericModel):
    """BLIP model for image-only classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_classes: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = BlipConfig.from_json_file(config_path)
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.vision_model = BlipVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, projection_dim, bias=False)
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.init_weights()

        if freeze_base_model:
            _freeze(self.vision_model)

        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.visual_projection(
            self.vision_model(pixel_values=pixel_values).pooler_output
        )
        return self.classifier(F.relu(image_embeds))


class BlipForImageCaption(GenericModel):
    """BLIP model for image captioning."""

    prefix_keys_in_state_dict = {"^(?!model\\.).*": "model."}

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = BlipConfig.from_json_file(config_path)
        self.config.vision_config.gradient_checkpointing = gradient_checkpointing
        self.config.text_config.gradient_checkpointing = gradient_checkpointing
        self.model = BlipForConditionalGeneration(self.config)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).decoder_logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 5,
        decoder_start_token_id: int = 101,
        decoder_end_token_id: Union[int, List[int]] = 102,
        num_return_sequences: int = 1,
        min_gen_seq_length: int = 0,
        max_gen_seq_length: int = 48,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
        length_penalty: float = 1.0,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> GenericOutputs:
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_seq_length,
            min_length=min_gen_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences.reshape(-1, num_return_sequences, outputs.sequences.size(-1))
        padded = torch.full(
            (sequences.size(0), num_return_sequences, max_gen_seq_length),
            fill_value=decoder_start_token_id,
            device=sequences.device,
            dtype=sequences.dtype,
        )
        padded[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            padded = padded.reshape(-1, max_gen_seq_length)

        return GenericOutputs(sequences=padded, sequences_scores=outputs.sequences_scores)
