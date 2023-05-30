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
from transformers.models.swin import SwinConfig, SwinModel
from unitorch.models import GenericModel


class SwinForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        """
        Initializes a SwinForImageClassification model for image classification tasks.

        Args:
            config_path (str): The path to the Swin Transformer configuration file.
            num_classes (int, optional): The number of classes for classification. Defaults to 1.
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
        Performs a forward pass of the SwinForImageClassification model.

        Args:
            pixel_values (torch.Tensor): The input pixel values of the image.

        Returns:
            (torch.Tensor):The model's logits.
        """
        outputs = self.swin(
            pixel_values=pixel_values,
        )

        pooled_output = outputs[1]
        return self.classifier(pooled_output)
