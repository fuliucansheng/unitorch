# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional
from transformers import DPTImageProcessor
from unitorch.models import HfImageClassificationProcessor


class DPTProcessor(HfImageClassificationProcessor):
    """
    Processor for DPT-based depth estimation models.
    """

    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes the DPTProcessor.

        Args:
            vision_config_path (str): Path to the DPT image processor configuration file.
        """
        vision_processor = DPTImageProcessor.from_json_file(vision_config_path)
        super().__init__(vision_processor=vision_processor)
