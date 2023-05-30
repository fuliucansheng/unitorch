# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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
            config_path (str): Path to the configuration file.
            num_classes (Optional[int]): Number of classes. Defaults to 1.
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
            pixel_values (torch.Tensor): Input tensor of shape [batch_size, num_channels, height, width].

        Returns:
            (torch.Tensor):Output logits of shape [batch_size, num_classes].
        """
        vision_outputs = self.vit(
            pixel_values=pixel_values,
        )
        pooled_output = vision_outputs[1]
        return self.classifier(pooled_output)
