# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional
from transformers import SegformerImageProcessor
from unitorch.models import HfImageClassificationProcessor


class SegformerProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes the SegformerProcessor.

        Args:
            vision_config_path (str): Path to the SegformerImageProcessor configuration file.
        """
        vision_processor = SegformerImageProcessor.from_json_file(vision_config_path)
        super().__init__(
            vision_processor=vision_processor,
        )
