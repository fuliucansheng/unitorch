# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.dpt import (
    DPTConfig,
    DPTForDepthEstimation as _DPTForDepthEstimation,
)
from unitorch.models import GenericModel


class DPTForDepthEstimation(GenericModel):
    """
    DPT model for depth estimation tasks.
    """

    prefix_keys_in_state_dict = {
        "^dpt.": "_base_model.",
        "^backbone.": "_base_model.",
        "^neck.": "_base_model.",
        "^head.": "_base_model.",
    }

    def __init__(
        self,
        config_path: str,
    ):
        """
        Initializes the DPTForDepthEstimation model.

        Args:
            config_path (str): Path to the DPT configuration file.
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
        Forward pass of the DPTForDepthEstimation model.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Upsampled depth prediction tensor.
        """
        predicted_depth = self._base_model(pixel_values=pixel_values).predicted_depth
        prediction = nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            scale_factor=4,
            mode="bicubic",
            align_corners=False,
        )
        return prediction
