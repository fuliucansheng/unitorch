# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_vit_infos = {
    "default-vit": {
        "config": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/preprocessor_config.json",
    },
    "vit-base-patch16-224-in21k": {
        "config": "https://huggingface.co/google/vit-base-patch16-224-in21k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-base-patch16-224-in21k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-base-patch16-224-in21k/resolve/main/pytorch_model.bin",
    },
    "vit-base-patch32-224-in21k": {
        "config": "https://huggingface.co/google/vit-base-patch32-224-in21k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-base-patch32-224-in21k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-base-patch32-224-in21k/resolve/main/pytorch_model.bin",
    },
    "vit-large-patch16-224-in21k": {
        "config": "https://huggingface.co/google/vit-large-patch16-224-in21k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-large-patch16-224-in21k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-large-patch16-224-in21k/resolve/main/pytorch_model.bin",
    },
    "vit-large-patch32-224-in21k": {
        "config": "https://huggingface.co/google/vit-large-patch32-224-in21k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-large-patch32-224-in21k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-large-patch32-224-in21k/resolve/main/pytorch_model.bin",
    },
    "vit-huge-patch14-224-in21k": {
        "config": "https://huggingface.co/google/vit-huge-patch14-224-in21k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-huge-patch14-224-in21k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-huge-patch14-224-in21k/resolve/main/pytorch_model.bin",
    },
    "vit-base-patch16-224": {
        "config": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/pytorch_model.bin",
    },
    "vit-base-patch16-384": {
        "config": "https://huggingface.co/google/vit-base-patch16-384/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-base-patch16-384/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-base-patch16-384/resolve/main/pytorch_model.bin",
    },
    "vit-base-patch32-384": {
        "config": "https://huggingface.co/google/vit-base-patch32-384/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-base-patch32-384/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-base-patch32-384/resolve/main/pytorch_model.bin",
    },
    "vit-large-patch16-224": {
        "config": "https://huggingface.co/google/vit-large-patch16-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-large-patch16-224/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-large-patch16-224/resolve/main/pytorch_model.bin",
    },
    "vit-large-patch16-384": {
        "config": "https://huggingface.co/google/vit-large-patch16-384/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-large-patch16-384/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-large-patch16-384/resolve/main/pytorch_model.bin",
    },
    "vit-large-patch32-384": {
        "config": "https://huggingface.co/google/vit-large-patch32-384/resolve/main/config.json",
        "vision_config": "https://huggingface.co/google/vit-large-patch32-384/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/google/vit-large-patch32-384/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.vit.modeling
import unitorch.cli.models.vit.processing
from unitorch.cli.models.vit.modeling import ViTForImageClassification
from unitorch.cli.models.vit.processing import ViTProcessor
