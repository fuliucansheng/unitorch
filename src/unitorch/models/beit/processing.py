# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from transformers import BeitImageProcessor

from unitorch.models import HfImageClassificationProcessor


class BeitProcessor(HfImageClassificationProcessor):
    """Image classification processor for BEiT models."""

    def __init__(self, vision_config_path: str) -> None:
        super().__init__(
            vision_processor=BeitImageProcessor.from_json_file(vision_config_path),
        )
