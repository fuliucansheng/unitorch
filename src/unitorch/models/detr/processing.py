# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessorPil

from unitorch.models import GenericOutputs


class DetrProcessor:
    """Image processor for DETR detection and segmentation models."""

    def __init__(
        self,
        vision_config_path: str,
        min_size_test: int = 800,
        max_size_test: int = 1333,
    ) -> None:
        self.vision_processor = DetrImageProcessorPil.from_json_file(vision_config_path)
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test

    def image(self, image: Image.Image) -> GenericOutputs:
        """Preprocess a single image and return pixel values with original size."""
        width, height = image.size
        pixel_values = self.vision_processor.preprocess(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)
        return GenericOutputs(image=pixel_values, sizes=torch.tensor([height, width]))

    def detection(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        classes: List[int],
    ) -> GenericOutputs:
        """Preprocess an image and normalise bounding boxes for detection training."""
        outputs = self.image(image)
        org_h, org_w = outputs.sizes
        bboxes = torch.tensor(bboxes, dtype=torch.float)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)
        scale = torch.tensor([org_w, org_h, org_w, org_h], dtype=torch.float)
        bboxes = bboxes / scale
        classes = torch.tensor(classes)
        assert bboxes.size(-1) == 4 and classes.dim() == 1 and len(classes) == len(bboxes)
        return GenericOutputs(image=outputs.image, bboxes=bboxes, classes=classes)

    def segmentation(
        self,
        image: Image.Image,
        gt_image: Image.Image,
        num_classes: Optional[int] = None,
    ) -> GenericOutputs:
        """Preprocess an image and its ground-truth segmentation mask."""
        pixel_values = self.image(image).image
        gt = np.array(gt_image)
        if num_classes is not None:
            gt = np.minimum(gt, num_classes)
        return GenericOutputs(image=pixel_values, gt_image=torch.tensor(gt))
