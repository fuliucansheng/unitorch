# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from transformers.models.sam.modeling_sam import SamConfig, SamModel
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin


class SamForSegmentation(GenericModel, PeftWeightLoaderMixin):
    """
    SAM model for segmentation tasks.
    """

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
        Initializes the SamForSegmentation model.

        Args:
            config_path (str): Path to the SAM configuration file.
        """
        super().__init__()
        config = SamConfig.from_json_file(config_path)
        self.sam = SamModel(config)
        self.init_weights()

    def forward(self):
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
        Runs segmentation inference.

        Args:
            pixel_values (torch.Tensor): Input image pixel values.
            input_points (torch.Tensor): Input point prompts.
            input_labels (torch.Tensor, optional): Labels for input points. Defaults to None.
            input_boxes (torch.Tensor, optional): Input box prompts. Defaults to None.
            input_masks (torch.Tensor, optional): Input mask prompts. Defaults to None.

        Returns:
            GenericOutputs: Predicted masks and IoU scores.
        """
        outputs = self.sam(
            pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            input_masks=input_masks,
        )
        return GenericOutputs(masks=outputs.pred_masks, scores=outputs.iou_scores)
