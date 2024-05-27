# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Optional
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import Mask2FormerImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
from unitorch.models import HfImageClassificationProcessor, GenericOutputs


class Mask2FormerProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes a Mask2FormerProcessor for image classification tasks.

        Args:
            vision_config_path (str): The path to the Mask2FormerImageProcessor configuration file.
        """
        vision_processor = Mask2FormerImageProcessor.from_json_file(vision_config_path)

        super().__init__(
            vision_processor=vision_processor,
        )
