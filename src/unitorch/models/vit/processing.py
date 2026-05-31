# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional
from transformers import ViTImageProcessor
from unitorch.models import HfImageClassificationProcessor


class ViTProcessor(HfImageClassificationProcessor):
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
            vision_config_path (str): Path to the ViT image processor configuration file.
        """
        vision_processor = ViTImageProcessor.from_json_file(vision_config_path)
        super().__init__(vision_processor=vision_processor)
