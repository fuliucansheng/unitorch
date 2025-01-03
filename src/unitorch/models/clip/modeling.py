# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.models.clip.modeling_clip import (
    CLIPConfig,
    CLIPTextTransformer,
    CLIPVisionTransformer,
)
from unitorch.models import GenericModel
from unitorch.models.peft import PeftWeightLoaderMixin


def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = _contrastive_loss(similarity)
    image_loss = _contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class AllGather(torch.autograd.Function):
    """
    Custom autograd function to perform all-gather operation on tensors.
    """

    @staticmethod
    def forward(ctx, tensor):
        """
        Forward pass of the all-gather operation.

        Args:
            ctx (torch.autograd.function.Context): Autograd function context.
            tensor (torch.Tensor): Input tensor to gather.

        Returns:
            (torch.Tensor):Gathered tensor.
        """
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        dist.all_gather(y, tensor)
        y = torch.cat(y, 0).view(-1, *tensor.size())
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the all-gather operation.

        Args:
            ctx (torch.autograd.function.Context): Autograd function context.
            grad_output (torch.Tensor): Gradient of the output.

        Returns:
            (torch.Tensor):Gradient of the input tensor.
        """
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        dist.all_reduce(in_grad)
        # split
        return in_grad[ctx.rank]


class ClipForPretrain(GenericModel):
    """
    Clip model for pretraining.
    """

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        """
        Initializes the ClipForPretrain model.

        Args:
            config_path (str): Path to the model configuration file.
            projection_dim (int, optional): Dimension of the projected embeddings. Defaults to 512.
            freeze_base_model (bool, optional): Whether to freeze the base model parameters. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_all_gather (bool, optional): Whether to use all-gather operation. Defaults to True.
        """
        super().__init__()

        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.use_all_gather = use_all_gather

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def _all_gather(self, input):
        """
        Perform all-gather operation on the input tensor.

        Args:
            input (torch.Tensor): Input tensor to gather.

        Returns:
            (torch.Tensor):Gathered tensor.
        """
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the Clip model.

        Args:
            input_ids (torch.Tensor, optional): Input text token IDs. Defaults to None.
            pixel_values (torch.Tensor, optional): Input image pixel values. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask for the input. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs for the input tokens. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.

        Returns:
            (torch.Tensor):Logits per text.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self._all_gather(text_embeds)
            image_embeds = self._all_gather(image_embeds)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        return _clip_loss(logits_per_text)


class ClipForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Clip model for classification.

        Args:
            config_path (str): Config file path to Clip model.
            projection_dim (int): Dimension for image/text output embedding.
            num_classes (int): Number of classes for classification.
            freeze_base_model (bool): Whether to freeze the base model.
            gradient_checkpointing (Optional[bool]): Whether to enable gradient_checkpointing.
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(self.projection_dim * 2, num_classes)

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
        Forward pass of the Clip model for classification.

        Args:
            input_ids (tensor): Tokens of text.
            pixel_values (tensor): Pixels of image.
            attention_mask (tensor): Attention mask of tokens.
            position_ids (tensor): Position IDs.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.

        Returns:
            tensor: Output tensor from the classifier.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        return self.classifier(F.relu(torch.cat([image_embeds, text_embeds], axis=1)))


class ClipForTextClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the Clip model for text classification.

        Args:
            config_path (str): The path to the CLIP configuration file.
            projection_dim (int, optional): The dimension of the projection layer. Defaults to 512.
            num_classes (int, optional): The number of classes for classification. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model parameters. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        text_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(self.projection_dim, num_classes)

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
        Forward pass of the Clip model for text classification.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position IDs.

        Returns:
            (torch.Tensor):The output logits.
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return self.classifier(F.relu(text_embeds))


class ClipForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the Clip model for image classification.

        Args:
            config_path (str): The path to the CLIP configuration file.
            projection_dim (int, optional): The dimension of the projection layer. Defaults to 512.
            num_classes (int, optional): The number of classes for classification. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model parameters. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.classifier = nn.Linear(self.projection_dim, num_classes)
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
        Forward pass of the Clip model for image classification.

        Args:
            pixel_values (torch.Tensor): The input pixel values.

        Returns:
            (torch.Tensor):The output logits.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return self.classifier(F.relu(image_embeds))


class ClipForMatching(GenericModel, PeftWeightLoaderMixin):
    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": ""}

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Clip model for classification.

        Args:
            config_path (str): Config file path to Clip model.
            projection_dim (int): Dimension for image/text output embedding.
            num_classes (int): Number of classes for classification.
            freeze_base_model (bool): Whether to freeze the base model.
            gradient_checkpointing (Optional[bool]): Whether to enable gradient_checkpointing.
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

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
        Forward pass of the Clip model for classification.

        Args:
            input_ids (tensor): Tokens of text.
            pixel_values (tensor): Pixels of image.
            attention_mask (tensor): Attention mask of tokens.
            position_ids (tensor): Position IDs.
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.

        Returns:
            tensor: Output tensor from the classifier.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return outputs
