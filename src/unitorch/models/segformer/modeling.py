# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
        Initializes a SamForSegmentation model for segmentation tasks.

        Args:
            config_path (str): The path to the Sam Transformer configuration file.
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
        Performs a forward pass of the SamForSegmentation model.
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
