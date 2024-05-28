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

from transformers.models.dpt import (
    DPTConfig,
    DPTForDepthEstimation as _DPTForDepthEstimation,
)
from unitorch.models import GenericModel


class DPTForDepthEstimation(GenericModel):
    prefix_keys_in_state_dict = {
        "^dpt.": "_base_model.",
        "^backbone.": "_base_model.",
        "^neck.": "_base_model.",
        "^head.": "_base_model.",
    }
    """
    ViT model for image classification tasks.
    """

    def __init__(
        self,
        config_path: str,
    ):
        """
        Initializes the ViTForImageClassification model.

        Args:
            config_path (str): Path to the configuration file.
            num_classes (Optional[int]): Number of classes. Defaults to 1.
        """
        super().__init__()
        config = DPTConfig.from_json_file(config_path)

        self._base_model = _DPTForDepthEstimation(config)
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
        predicted_depth = self._base_model(
            pixel_values=pixel_values,
        ).predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        )
        return prediction
