# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple, Union
from transformers import SamImageProcessor
from unitorch.models import GenericOutputs


class SamProcessor:
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes the SamProcessor.

        Args:
            vision_config_path (str): Path to the SamImageProcessor configuration file.
        """
        self.vision_processor = SamImageProcessor.from_json_file(vision_config_path)

    def segmentation_inputs(
        self,
        image: Union[Image.Image, str],
        crops_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[int] = 1,
    ):
        """
        Generates segmentation inputs using grid-based point prompts.

        Args:
            image (PIL.Image.Image or str): Input image or path.
            crops_n_layers (int, optional): Number of crop layers. Defaults to 0.
            crop_overlap_ratio (float, optional): Overlap ratio between crops. Defaults to 512/1500.
            points_per_crop (int, optional): Number of grid points per crop. Defaults to 32.
            crop_n_points_downscale_factor (int, optional): Downscale factor for points. Defaults to 1.

        Returns:
            GenericOutputs: Processed pixel values, grid points, labels, and boxes.
        """
        if isinstance(image, str):
            image = Image.open(image)

        target_size = self.vision_processor.size["longest_edge"]
        crop_boxes, grid_points, cropped_images, input_labels = (
            self.vision_processor.generate_crop_boxes(
                image,
                target_size,
                crops_n_layers,
                crop_overlap_ratio,
                points_per_crop,
                crop_n_points_downscale_factor,
            )
        )
        pixel_inputs = self.vision_processor(cropped_images, return_tensors="pt")
        return GenericOutputs(
            pixel_values=pixel_inputs.get("pixel_values")[0],
            original_sizes=pixel_inputs.get("original_sizes")[0],
            reshaped_input_sizes=pixel_inputs.get("reshaped_input_sizes")[0],
            input_points=grid_points[0],
            input_labels=input_labels[0],
            input_boxes=crop_boxes[0],
        )

    def processing_masks(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        original_sizes: Union[torch.Tensor, List[Tuple[int, int]]],
        reshaped_input_sizes: Union[torch.Tensor, List[Tuple[int, int]]],
        input_boxes: torch.Tensor,
        mask_threshold: Optional[float] = 0.0,
        pred_iou_thresh: Optional[float] = 0.88,
        stability_score_thresh: Optional[float] = 0.95,
        stability_score_offset: Optional[int] = 1,
        crops_nms_thresh: Optional[float] = 0.7,
    ):
        """
        Post-processes predicted masks and scores.

        Args:
            masks (torch.Tensor): Raw predicted masks.
            scores (torch.Tensor): IoU scores.
            original_sizes: Original image sizes.
            reshaped_input_sizes: Reshaped input sizes.
            input_boxes (torch.Tensor): Crop boxes.
            mask_threshold (float, optional): Threshold for mask binarization. Defaults to 0.0.
            pred_iou_thresh (float, optional): IoU score threshold. Defaults to 0.88.
            stability_score_thresh (float, optional): Stability score threshold. Defaults to 0.95.
            stability_score_offset (int, optional): Stability score offset. Defaults to 1.
            crops_nms_thresh (float, optional): NMS threshold for crops. Defaults to 0.7.

        Returns:
            GenericOutputs: Filtered masks, scores, RLE masks, and bounding boxes.
        """
        if isinstance(original_sizes, torch.Tensor):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, torch.Tensor):
            reshaped_input_sizes = reshaped_input_sizes.tolist()

        masks = self.vision_processor.post_process_masks(
            masks,
            original_sizes,
            reshaped_input_sizes,
            mask_threshold=mask_threshold,
            binarize=False,
        )

        output_masks, output_scores, output_rle_mask, output_bounding_boxes = [], [], [], []
        for _masks, _scores, _original_sizes, _input_boxes in zip(
            masks, scores, original_sizes, input_boxes
        ):
            _masks, _scores, _boxes = self.vision_processor.filter_masks(
                _masks,
                _scores,
                _original_sizes,
                _input_boxes,
                pred_iou_thresh,
                stability_score_thresh,
                mask_threshold,
                stability_score_offset,
            )
            _masks, _scores, _rle_mask, _bounding_boxes = (
                self.vision_processor.post_process_for_mask_generation(
                    _masks, _scores, _boxes, crops_nms_thresh
                )
            )
            output_masks.append(torch.from_numpy(np.array(_masks)))
            output_scores.append(_scores)
            output_rle_mask.append(_rle_mask)
            output_bounding_boxes.append(_bounding_boxes)

        return GenericOutputs(
            masks=output_masks,
            scores=output_scores,
            rle_mask=output_rle_mask,
            bounding_boxes=output_bounding_boxes,
        )
