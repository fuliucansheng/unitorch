# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import (
    CLIPConfig,
    CLIPTextModel,
    CLIPVisionModel,
)

from unitorch.models import GenericModel
from unitorch.models.clip.processing import ClipProcessor
from unitorch.models.peft import PeftWeightLoaderMixin


def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return (_contrastive_loss(similarity) + _contrastive_loss(similarity.T)) / 2.0


def _freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


class AllGather(torch.autograd.Function):
    """All-gather with gradient support for distributed contrastive training."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0).view(-1, *tensor.size())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        in_grad = grad_output.clone()
        dist.all_reduce(in_grad)
        return in_grad[ctx.rank]


class ClipForPretrain(GenericModel):
    """CLIP model for contrastive image-text pre-training."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
        use_all_gather: bool = True,
    ) -> None:
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.use_all_gather = use_all_gather
        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, projection_dim, bias=False
        )
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, projection_dim, bias=False
        )
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
            self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).pooler_output
        )
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        if self.use_all_gather and dist.is_initialized():
            text_embeds = self._all_gather(text_embeds)
            image_embeds = self._all_gather(image_embeds)

        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp()
        )
        return _clip_loss(logits_per_text)


class ClipForClassification(GenericModel):
    """CLIP model for multimodal (image + text) classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_classes: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, projection_dim, bias=False
        )
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, projection_dim, bias=False
        )
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
            self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).pooler_output
        )
        return self.classifier(F.relu(torch.cat([image_embeds, text_embeds], dim=1)))


class ClipForTextClassification(GenericModel):
    """CLIP model for text-only classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_classes: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing

        self.text_model = CLIPTextModel(config.text_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, projection_dim, bias=False
        )
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
            self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).pooler_output
        )
        return self.classifier(F.relu(text_embeds))


class ClipForImageClassification(GenericModel):
    """CLIP model for image-only classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        num_classes: int = 1,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, projection_dim, bias=False
        )
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


class ClipForImageClassificationV2(GenericModel, PeftWeightLoaderMixin):
    """CLIP model for prompt-based image classification with end-to-end finetuning."""

    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": ""}

    def __init__(
        self,
        config_path: str,
        labels: List[str],
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        projection_dim: Optional[int] = None,
        output_embed_dim: Optional[int] = None,
        max_seq_length: Optional[int] = 128,
        freeze_base_model: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        if projection_dim is None:
            projection_dim = config.projection_dim
        config.text_config.gradient_checkpointing = gradient_checkpointing
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, projection_dim, bias=False
        )
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, projection_dim, bias=False
        )
        self.output_projection = (
            nn.Linear(projection_dim, output_embed_dim)
            if output_embed_dim is not None
            else None
        )
        self.classifier = nn.Linear(1, 1)

        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        if labels is None or len(labels) == 0:
            raise ValueError("labels must be provided for ClipForImageClassificationV2")
        self.labels_inputs = self.get_label_inputs(labels)
        self.labels_embeds = None

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            _freeze(self.text_model)
            _freeze(self.vision_model)

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def train(self, mode: bool = True):
        if mode:
            self.labels_embeds = None
        return super().train(mode)

    def from_pretrained(self, *args, **kwargs):
        self.labels_embeds = None
        return super().from_pretrained(*args, **kwargs)

    def get_label_inputs(self, texts: List[str]):
        input_ids, attention_mask, position_ids = [], [], []
        for text in texts:
            inputs = self.processor.text_classification(text)
            input_ids.append(inputs.input_ids)
            attention_mask.append(inputs.attention_mask)
            position_ids.append(inputs.position_ids)

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "position_ids": torch.stack(position_ids, dim=0),
        }

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(
            self.vision_model(pixel_values=pixel_values).pooler_output
        )

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_projection(
            self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).pooler_output
        )

    def _project_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        if self.output_projection is not None:
            embeds = self.output_projection(embeds)
        return embeds

    def _normalize_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        return embeds / embeds.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def _get_label_embeds(self, device: torch.device) -> torch.Tensor:
        label_inputs = self.labels_inputs
        text_embeds = self.get_text_features(
            input_ids=label_inputs["input_ids"].to(device),
            attention_mask=label_inputs["attention_mask"].to(device),
            position_ids=label_inputs["position_ids"].to(device),
        )
        text_embeds = self._project_embeds(text_embeds)
        return self._normalize_embeds(text_embeds)

    def _get_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.get_image_features(pixel_values=pixel_values)
        image_embeds = self._project_embeds(image_embeds)
        return self._normalize_embeds(image_embeds)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.labels_embeds = None
            text_embeds = self._get_label_embeds(pixel_values.device)
        else:
            if self.labels_embeds is None or self.labels_embeds.device != pixel_values.device:
                self.labels_embeds = self._get_label_embeds(pixel_values.device)
            text_embeds = self.labels_embeds

        image_embeds = self._get_image_embeds(pixel_values)
        scores = torch.einsum("bd,cd->bc", image_embeds, text_embeds)
        return self.classifier(scores.view(-1, 1)).view(-1, text_embeds.size(0))


class ClipForMatching(GenericModel, PeftWeightLoaderMixin):
    """CLIP model for image-text matching (cosine similarity scoring)."""

    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": ""}

    def __init__(
        self,
        config_path: str,
        projection_dim: int = 512,
        freeze_base_model: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        config.text_config.gradient_checkpointing = gradient_checkpointing
        config.vision_config.gradient_checkpointing = gradient_checkpointing

        self.text_model = CLIPTextModel(config.text_config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, projection_dim, bias=False
        )
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, projection_dim, bias=False
        )
        self.classifier = nn.Linear(1, 1)
        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

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
            self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).pooler_output
        )
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)
        return self.classifier(scores)
