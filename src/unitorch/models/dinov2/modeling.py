# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.dinov2 import Dinov2Config, Dinov2Model
from unitorch.models import GenericModel


class DinoV2ForImageClassification(GenericModel):
    """
    DINOv2 model for image classification tasks.
    """

    prefix_keys_in_state_dict = {
        "^embeddings.": "dinov2.",
        "^layernorm.": "dinov2.",
        "^encoder.": "dinov2.",
    }

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        """
        Initializes the DinoV2ForImageClassification model.

        Args:
            config_path (str): Path to the DINOv2 configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
        """
        super().__init__()
        config = Dinov2Config.from_json_file(config_path)
        self.dinov2 = Dinov2Model(config)
        self.classifier = nn.Linear(config.hidden_size * 2, num_classes)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the DinoV2ForImageClassification model.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Classification logits of shape [batch_size, num_classes].
        """
        vision_outputs = self.dinov2(pixel_values=pixel_values)[0]
        cls_output = vision_outputs[:, 0]
        patch_output = vision_outputs[:, 1:]
        pooled_output = torch.cat([cls_output, patch_output.mean(dim=1)], dim=-1)
        return self.classifier(pooled_output)
