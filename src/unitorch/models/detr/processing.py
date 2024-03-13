# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from hmac import new
import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import DetrFeatureExtractor
from unitorch.utils import resize_shortest_edge, pop_value
from unitorch.models import GenericOutputs


class DetrProcessor(object):
    def __init__(
        self,
        vision_config_path: str,
        min_size_test: Optional[int] = 800,
        max_size_test: Optional[int] = 1333,
    ):
        """
        Args:
            vision_config_path: vision config path to detr processor
            min_size_test: resize shortest edge parameters
            max_size_test: resize shortest edge parameters
        """
        self.vision_processor = DetrFeatureExtractor.from_json_file(vision_config_path)

        self.image_mean = self.vision_processor.image_mean
        self.image_std = self.vision_processor.image_std
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test

    def image(
        self,
        image: Image.Image,
    ):
        """
        Args:
            image: input image
        """
        width, height = image.size
        image = resize_shortest_edge(
            image.convert("RGB"),
            [self.min_size_test, self.max_size_test],
            self.max_size_test,
        )

        image = np.array(image) / 255.0

        image = self.vision_processor.normalize(
            image=image,
            mean=self.image_mean,
            std=self.image_std,
        )
        return GenericOutputs(
            image=torch.tensor(image).permute(2, 0, 1),
            sizes=torch.tensor([height, width]),
        )

    def detection(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        classes: List[int],
    ):
        """
        Args:
            image: input image
            bboxes: bboxes to image
            classes: class to each bbox
        """
        outputs = self.image(image)
        # new_h, new_w = outputs.image.size()[1:]
        org_h, org_w = outputs.sizes
        image = outputs.image
        bboxes = torch.tensor(bboxes).float()
        bboxes[:, 0] = bboxes[:, 0] / org_w
        bboxes[:, 1] = bboxes[:, 1] / org_h
        bboxes[:, 2] = bboxes[:, 2] / org_w
        bboxes[:, 3] = bboxes[:, 3] / org_h
        classes = torch.tensor(classes)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)

        assert (
            bboxes.size(-1) == 4 and classes.dim() == 1 and len(classes) == len(bboxes)
        )
        return GenericOutputs(image=image, bboxes=bboxes, classes=classes)

    def segmentation(
        self,
        image: Image.Image,
        gt_image: Image.Image,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            image: input image
            gt_image: ground truth image
            num_classes: num classes to classification
        """
        image = self.image(image).image
        gt_image = np.array(gt_image)
        if num_classes is not None:
            gt_image = np.minimum(gt_image, num_classes)
        return GenericOutputs(
            image=torch.tensor(image),
            gt_image=torch.tensor(gt_image),
        )
