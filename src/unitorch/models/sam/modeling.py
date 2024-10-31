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
from transformers.models.sam.modeling_sam import SamConfig, SamModel
from transformers.models.sam.image_processing_sam import _build_point_grid
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin


class SamForSegmentation(GenericModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^mask_decoder.*": "sam.",
        "^vision_encoder.*": "sam.",
        "^prompt_encoder.*": "sam.",
        "^shared_image_embedding.*": "sam.",
    }

    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": ""}

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
        config = SamConfig.from_json_file(config_path)

        self.sam = SamModel(config)
        self.init_weights()

    def forward(self):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        raise NotImplementedError

    def segment(
        self,
        pixel_values: torch.Tensor,
        input_points: torch.Tensor,
        input_labels: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        outputs = self.sam(
            pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            # input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return GenericOutputs(masks=outputs.pred_masks, scores=outputs.iou_scores)
