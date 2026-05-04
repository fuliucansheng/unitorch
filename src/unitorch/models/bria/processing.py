# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import torch
from unitorch.models import GenericOutputs


class BRIAProcessor:
    """Image processor for BRIA segmentation models."""

    def __init__(self, image_size: int = 1024) -> None:
        self.image_size = image_size
        self.transform_inputs = Compose([
            Resize([image_size, image_size]),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_labels = Compose([
            Resize([image_size, image_size]),
            ToTensor(),
        ])

    def segmentation_inputs(self, image: Image.Image) -> GenericOutputs:
        """Preprocess an RGB image as model input."""
        width, height = image.size
        return GenericOutputs(
            image=self.transform_inputs(image.convert("RGB")),
            sizes=torch.tensor([height, width]),
        )

    def segmentation_labels(self, image: Image.Image) -> GenericOutputs:
        """Preprocess a grayscale mask as segmentation label."""
        width, height = image.size
        return GenericOutputs(
            image=self.transform_labels(image.convert("L")),
            sizes=torch.tensor([height, width]),
        )
