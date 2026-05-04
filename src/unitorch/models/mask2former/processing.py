# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from transformers import Mask2FormerImageProcessor
from unitorch.models import HfImageClassificationProcessor, GenericOutputs


class Mask2FormerProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
    ):
        vision_processor = Mask2FormerImageProcessor.from_json_file(vision_config_path)

        super().__init__(
            vision_processor=vision_processor,
        )
