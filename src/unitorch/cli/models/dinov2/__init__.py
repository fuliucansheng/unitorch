# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_dinov2_infos = {
    "default-dinov2": {
        "config": "https://huggingface.co/facebook/dinov2-small/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/dinov2-small/resolve/main/preprocessor_config.json",
    },
    "dinov2-base": {
        "config": "https://huggingface.co/facebook/dinov2-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/dinov2-base/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/dinov2-base/resolve/main/pytorch_model.bin",
    },
    "dinov2-small": {
        "config": "https://huggingface.co/facebook/dinov2-small/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/dinov2-small/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/dinov2-small/resolve/main/pytorch_model.bin",
    },
    "dinov2-large": {
        "config": "https://huggingface.co/facebook/dinov2-large/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/dinov2-large/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/dinov2-large/resolve/main/pytorch_model.bin",
    },
    "dinov2-giant": {
        "config": "https://huggingface.co/facebook/dinov2-giant/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/dinov2-giant/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/dinov2-giant/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.dinov2.modeling
import unitorch.cli.models.dinov2.processing
from unitorch.cli.models.dinov2.modeling import DinoV2ForImageClassification
from unitorch.cli.models.dinov2.processing import DinoV2Processor
