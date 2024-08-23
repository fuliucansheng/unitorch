# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Optional
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import SegformerImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value
from unitorch.models import HfImageClassificationProcessor, GenericOutputs


class SegformerProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes a SegformerProcessor for image classification tasks.

        Args:
            vision_config_path (str): The path to the SegformerImageProcessor configuration file.
        """
        vision_processor = SegformerImageProcessor.from_json_file(vision_config_path)

        super().__init__(
            vision_processor=vision_processor,
        )
