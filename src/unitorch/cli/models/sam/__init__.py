# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_sam_infos = {
    "default-sam": {
        "config": "https://huggingface.co/facebook/sam-vit-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/sam-vit-base/resolve/main/preprocessor_config.json",
    },
    "sam-vit-base": {
        "config": "https://huggingface.co/facebook/sam-vit-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/sam-vit-base/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/sam-vit-base/resolve/main/pytorch_model.bin",
    },
    "sam-vit-large": {
        "config": "https://huggingface.co/facebook/sam-vit-large/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/sam-vit-large/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/sam-vit-large/resolve/main/pytorch_model.bin",
    },
    "sam-vit-huge": {
        "config": "https://huggingface.co/facebook/sam-vit-huge/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/sam-vit-huge/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/sam-vit-huge/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.sam.modeling
import unitorch.cli.models.sam.processing
from unitorch.cli.models.sam.modeling import SamForSegmentation
from unitorch.cli.models.sam.processing import SamProcessor
