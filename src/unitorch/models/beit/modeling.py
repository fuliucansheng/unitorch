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
from transformers.models.beit import BeitConfig, BeitModel
from unitorch.models import GenericModel


class BeitForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        """
        Initializes the BeitForImageClassification model.

        Args:
            config_path (str): The path to the configuration file.
            num_classes (Optional[int], optional): The number of classes for classification. Defaults to 1.
        """
        super().__init__()
        config = BeitConfig.from_json_file(config_path)

        self.beit = BeitModel(config, add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the BeitForImageClassification model.

        Args:
            pixel_values (torch.Tensor): The input tensor of pixel values.

        Returns:
            (torch.Tensor):The logits of the model output.
        """
        outputs = self.beit(
            pixel_values=pixel_values,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits
