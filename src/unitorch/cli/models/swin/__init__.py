# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_swin_infos = {
    "default-swin": {
        "config": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/preprocessor_config.json",
    },
    "swin-tiny-patch4-window7-224": {
        "config": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/pytorch_model.bin",
    },
    "swin-base-patch4-window7-224": {
        "config": "https://huggingface.co/microsoft/swin-base-patch4-window7-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-base-patch4-window7-224/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/swin-base-patch4-window7-224/resolve/main/pytorch_model.bin",
    },
    "swin-large-patch4-window7-224": {
        "config": "https://huggingface.co/microsoft/swin-large-patch4-window7-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-large-patch4-window7-224/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/swin-large-patch4-window7-224/resolve/main/pytorch_model.bin",
    },
    "swin-large-patch4-window12-384": {
        "config": "https://huggingface.co/microsoft/swin-large-patch4-window12-384/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-large-patch4-window12-384/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/swin-large-patch4-window12-384/resolve/main/pytorch_model.bin",
    },
    "swin-base-patch4-window7-224-in22k": {
        "config": "https://huggingface.co/microsoft/swin-base-patch4-window7-224-in22k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-base-patch4-window7-224-in22k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/swin-base-patch4-window7-224-in22k/resolve/main/pytorch_model.bin",
    },
    "swin-large-patch4-window7-224-in22k": {
        "config": "https://huggingface.co/microsoft/swin-large-patch4-window7-224-in22k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/swin-large-patch4-window7-224-in22k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/swin-large-patch4-window7-224-in22k/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.swin.modeling
import unitorch.cli.models.swin.processing
from unitorch.cli.models.swin.modeling import SwinForImageClassification
from unitorch.cli.models.swin.processing import SwinProcessor
