# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_blip_infos = {
    "blip-vqa-base": {
        "config": hf_endpoint_url("/Salesforce/blip-vqa-base/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/Salesforce/blip-vqa-base/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url("/Salesforce/blip-vqa-base/resolve/main/vocab.txt"),
        "weight": hf_endpoint_url(
            "/Salesforce/blip-vqa-base/resolve/main/pytorch_model.bin"
        ),
    },
    "blip-image-captioning-base": {
        "config": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-base/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-base/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-base/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-base/resolve/main/pytorch_model.bin"
        ),
    },
    "blip-image-captioning-large": {
        "config": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-large/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-large/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-large/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/Salesforce/blip-image-captioning-large/resolve/main/pytorch_model.bin"
        ),
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
