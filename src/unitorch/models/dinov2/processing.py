# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional
from transformers import BitImageProcessor
from unitorch.models import HfImageClassificationProcessor


class DinoV2Processor(HfImageClassificationProcessor):
    """
    Processor for DINOv2-based image classification models.
    """

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes the DinoV2Processor.

        Args:
            vision_config_path (str): Path to the vision processor configuration file.
        """
        vision_processor = BitImageProcessor.from_json_file(vision_config_path)
        super().__init__(vision_processor=vision_processor)
