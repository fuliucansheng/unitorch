# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.vit import ViTConfig, ViTModel
from unitorch.models import GenericModel


class ViTForImageClassification(GenericModel):
    """
    ViT model for image classification tasks.
    """

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        """
        Initializes the ViTForImageClassification model.

        Args:
            config_path (str): Path to the ViT configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
        """
        super().__init__()
        config = ViTConfig.from_json_file(config_path)
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the ViTForImageClassification model.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Classification logits of shape [batch_size, num_classes].
        """
        vision_outputs = self.vit(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]
        return self.classifier(pooled_output)
