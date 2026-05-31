# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.swin import SwinConfig, SwinModel
from unitorch.models import GenericModel


class SwinForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        """
        Initializes the SwinForImageClassification model.

        Args:
            config_path (str): Path to the Swin Transformer configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
        """
        super().__init__()
        config = SwinConfig.from_json_file(config_path)
        self.swin = SwinModel(config)
        self.classifier = nn.Linear(self.swin.num_features, num_classes)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the SwinForImageClassification model.

        Args:
            pixel_values (torch.Tensor): Input image pixel values.

        Returns:
            torch.Tensor: Classification logits.
        """
        outputs = self.swin(pixel_values=pixel_values)
        pooled_output = outputs[1]
        return self.classifier(pooled_output)
