# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_blip_infos = {
    "default-blip": {
        "config": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/vocab.txt",
    },
    "blip-vqa-base": {
        "config": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/pytorch_model.bin",
    },
    "blip-image-captioning-base": {
        "config": "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/config.json",
        "vision_config": "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin",
    },
    "blip-image-captioning-large": {
        "config": "https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/config.json",
        "vision_config": "https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.blip.modeling
import unitorch.cli.models.blip.processing
from unitorch.cli.models.blip.modeling import (
    BlipForPretrain,
    BlipForClassification,
    BlipForTextClassification,
    BlipForImageClassification,
    BlipForImageCaption,
)
from unitorch.cli.models.blip.processing import BlipProcessor
