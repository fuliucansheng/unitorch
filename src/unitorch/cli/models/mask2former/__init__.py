# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_mask2former_infos = {
    "default-mask2former": {
        "config": "https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/resolve/main/preprocessor_config.json",
    },
    "mask2former-swin-tiny-ade-semantic": {
        "config": "https://huggingface.co/facebook/mask2former-swin-tiny-ade-semantic/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/mask2former-swin-tiny-ade-semantic/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/mask2former-swin-tiny-ade-semantic/resolve/main/pytorch_model.bin",
    },
    "mask2former-swin-base-ade-semantic": {
        "config": "https://huggingface.co/facebook/mask2former-swin-base-ade-semantic/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/mask2former-swin-base-ade-semantic/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/mask2former-swin-base-ade-semantic/resolve/main/pytorch_model.bin",
    },
    "mask2former-swin-large-ade-semantic": {
        "config": "https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/resolve/main/config.json",
        "vision_config": "https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.mask2former.modeling
import unitorch.cli.models.mask2former.processing
from unitorch.cli.models.mask2former.modeling import Mask2FormerForSegmentation
from unitorch.cli.models.mask2former.processing import Mask2FormerProcessor
