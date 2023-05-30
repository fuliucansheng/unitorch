# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import ViTImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
from unitorch.models import HfImageClassificationProcessor, GenericOutputs


class SwinProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes a SwinProcessor for image classification tasks.

        Args:
            vision_config_path (str): The path to the ViTImageProcessor configuration file.
        """
        vision_processor = ViTImageProcessor.from_json_file(vision_config_path)
        super().__init__(
            vision_processor=vision_processor,
        )
