# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerModel,
    Mask2FormerLoss,
)
from unitorch.models import GenericModel, GenericOutputs


class Mask2FormerForSegmentation(GenericModel):
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
        config = Mask2FormerConfig.from_json_file(config_path)

        self.model = Mask2FormerModel(config)
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.init_weights()

    def forward(self):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        raise NotImplementedError

    def segment(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True,
        )

        decoder_output = outputs.transformer_decoder_intermediate_states[-1]
        class_queries_logits = self.class_predictor(decoder_output.transpose(0, 1))

        masks_queries_logits = outputs.masks_queries_logits[-1]
        masks_queries_logits = nn.functional.interpolate(
            masks_queries_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        return GenericOutputs(masks=masks_queries_logits, classes=class_queries_logits)
