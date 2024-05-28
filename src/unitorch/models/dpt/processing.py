# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import DPTImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
from unitorch.models import HfImageClassificationProcessor, GenericOutputs


class DPTProcessor(HfImageClassificationProcessor):
    """
    Processor for ViT-based image classification models.
    """

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes the ViTProcessor.

        Args:
            vision_config_path (str): Path to the vision configuration file.
        """
        vision_processor = DPTImageProcessor.from_json_file(vision_config_path)
        super().__init__(
            vision_processor=vision_processor,
        )
