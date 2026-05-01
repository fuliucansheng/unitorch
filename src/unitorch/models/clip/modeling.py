# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

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
        self.text_projection = nn.Linear(config.text_config.hidden_size, projection_dim, bias=False)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, projection_dim, bias=False)
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
            self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).pooler_output
        )
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)
        return self.classifier(scores)
