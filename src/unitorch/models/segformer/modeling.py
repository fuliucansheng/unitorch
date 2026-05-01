# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.segformer.modeling_segformer import (
    SegformerConfig,
    SegformerModel,
    SegformerDecodeHead,
)
from unitorch.models import GenericModel, GenericOutputs


class SegformerForSegmentation(GenericModel):
    def __init__(
        self,
        config_path: str,
    ):
        """
        Initializes the SegformerForSegmentation model.

        Args:
            config_path (str): Path to the Segformer configuration file.
        """
        super().__init__()
        config = SegformerConfig.from_json_file(config_path)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the SegformerForSegmentation model.

        Args:
            pixel_values (torch.Tensor): Input image pixel values.

        Returns:
            GenericOutputs: Object containing upsampled segmentation logits.
        """
        outputs = self.segformer(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.decode_head(outputs.hidden_states)
        upsampled_logits = nn.functional.interpolate(
            logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        return GenericOutputs(logits=upsampled_logits)
