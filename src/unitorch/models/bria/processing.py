# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from PIL import Image
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from unitorch.models import GenericOutputs


class BRIAProcessor:
    def __init__(
        self,
        image_size: Optional[int] = 1024,
    ):
        self.image_size = image_size
        self.transform_inputs = Compose(
            [
                Resize(size=[self.image_size, self.image_size]),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
            ]
        )
        self.transform_labels = Compose(
            [
                Resize(size=[self.image_size, self.image_size]),
                ToTensor(),
            ]
        )

    def segmentation_inputs(self, image: Image):
        width, height = image.size
        image = self.transform_inputs(image.convert("RGB"))
        return GenericOutputs(
            image=torch.tensor(image),
            sizes=torch.tensor([height, width]),
        )

    def segmentation_labels(self, image: Image):
        width, height = image.size
        image = self.transform_labels(image.convert("L"))
        return GenericOutputs(
            image=torch.tensor(image),
            sizes=torch.tensor([height, width]),
        )
