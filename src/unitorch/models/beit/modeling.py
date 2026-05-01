# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from transformers.models.beit import BeitConfig, BeitModel

from unitorch.models import GenericModel


class BeitForImageClassification(GenericModel):
    """BEiT model for image classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        config = BeitConfig.from_json_file(config_path)
        self.beit = BeitModel(config, add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.init_weights()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.beit(pixel_values=pixel_values).pooler_output)
