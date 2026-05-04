# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from transformers.models.siglip.modeling_siglip import (
    SiglipConfig,
    SiglipTextModel,
    SiglipVisionModel,
)
from unitorch.models import GenericModel
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.clip.modeling import _clip_loss, AllGather


class SiglipForPretrain(GenericModel):
    """
    Siglip model for pretraining.
    """

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        """
        Initializes the SiglipForPretrain model.

        Args:
            config_path (str): Path to the model configuration file.
            freeze_base_model (bool, optional): Whether to freeze the base model parameters. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_all_gather (bool, optional): Whether to use all-gather for distributed training. Defaults to True.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing
        vision_config.vision_use_head = True

        self.use_all_gather = use_all_gather
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = SiglipTextModel(text_config)
        self.vision_model = SiglipVisionModel(vision_config)
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def _all_gather(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs all-gather on the input tensor across distributed processes.

        Args:
            input (torch.Tensor): Input tensor to gather.

        Returns:
            torch.Tensor: Gathered tensor.
        """
        output = AllGather.apply(input)
        return output.view(-1, *(output.shape[2:]))

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the SiglipForPretrain model.

        Args:
            input_ids (torch.Tensor): Input text token IDs.
            pixel_values (torch.Tensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for the input.
            position_ids (torch.Tensor): Position IDs for the input tokens.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self._all_gather(text_embeds)
            image_embeds = self._all_gather(image_embeds)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        return _clip_loss(logits_per_text)


class SiglipForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Siglip model for multimodal classification.

        Args:
            config_path (str): Path to the Siglip configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing
        vision_config.vision_use_head = True

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = SiglipTextModel(text_config)
        self.vision_model = SiglipVisionModel(vision_config)
        self.classifier = nn.Linear(
            self.text_embed_dim + self.vision_embed_dim, num_classes
        )

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the SiglipForClassification model.

        Args:
            input_ids (torch.Tensor): Input text token IDs.
            pixel_values (torch.Tensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for the input.
            position_ids (torch.Tensor): Position IDs for the input tokens.

        Returns:
            torch.Tensor: Classification logits.
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        return self.classifier(F.relu(torch.cat([image_embeds, text_embeds], dim=1)))


class SiglipForTextClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Siglip model for text classification.

        Args:
            config_path (str): Path to the Siglip configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        text_config = config.text_config
        text_config.gradient_checkpointing = gradient_checkpointing

        self.text_embed_dim = text_config.hidden_size
        self.text_model = SiglipTextModel(text_config)
        self.classifier = nn.Linear(self.text_embed_dim, num_classes)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the SiglipForTextClassification model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            position_ids (torch.Tensor): Position IDs.

        Returns:
            torch.Tensor: Classification logits.
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_embeds = text_outputs[1]
        return self.classifier(F.relu(text_embeds))


class SiglipForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Siglip model for image classification.

        Args:
            config_path (str): Path to the Siglip configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing
        vision_config.vision_use_head = True

        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = SiglipVisionModel(vision_config)
        self.classifier = nn.Linear(self.vision_embed_dim, num_classes)

        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the SiglipForImageClassification model.

        Args:
            pixel_values (torch.Tensor): Input image pixel values.

        Returns:
            torch.Tensor: Classification logits.
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[1]
        return self.classifier(F.relu(image_embeds))


class SiglipForMatching(GenericModel, PeftWeightLoaderMixin):
    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": ""}

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Siglip model for image-text matching.

        Args:
            config_path (str): Path to the Siglip configuration file.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing
        vision_config.vision_use_head = True

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        assert self.text_embed_dim == self.vision_embed_dim

        self.text_model = SiglipTextModel(text_config)
        self.vision_model = SiglipVisionModel(vision_config)
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the SiglipForMatching model.

        Args:
            input_ids (torch.Tensor): Input text token IDs.
            pixel_values (torch.Tensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for the input.
            position_ids (torch.Tensor): Position IDs for the input tokens.

        Returns:
            torch.Tensor: Matching scores.
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)
        return self.classifier(scores)
