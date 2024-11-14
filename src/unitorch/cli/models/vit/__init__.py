# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_vit_infos = {
    "vit-base-patch16-224-in21k": {
        "config": hf_endpoint_url(
            "/google/vit-base-patch16-224-in21k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-base-patch16-224-in21k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-base-patch16-224-in21k/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-base-patch32-224-in21k": {
        "config": hf_endpoint_url(
            "/google/vit-base-patch32-224-in21k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-base-patch32-224-in21k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-base-patch32-224-in21k/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-large-patch16-224-in21k": {
        "config": hf_endpoint_url(
            "/google/vit-large-patch16-224-in21k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-large-patch16-224-in21k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-large-patch16-224-in21k/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-large-patch32-224-in21k": {
        "config": hf_endpoint_url(
            "/google/vit-large-patch32-224-in21k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-large-patch32-224-in21k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-large-patch32-224-in21k/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-huge-patch14-224-in21k": {
        "config": hf_endpoint_url(
            "/google/vit-huge-patch14-224-in21k/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-huge-patch14-224-in21k/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-huge-patch14-224-in21k/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-base-patch16-224": {
        "config": hf_endpoint_url(
            "/google/vit-base-patch16-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-base-patch16-224/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-base-patch16-224/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-base-patch16-384": {
        "config": hf_endpoint_url(
            "/google/vit-base-patch16-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-base-patch16-384/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-base-patch16-384/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-base-patch32-384": {
        "config": hf_endpoint_url(
            "/google/vit-base-patch32-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-base-patch32-384/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-base-patch32-384/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-large-patch16-224": {
        "config": hf_endpoint_url(
            "/google/vit-large-patch16-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-large-patch16-224/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-large-patch16-224/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-large-patch16-384": {
        "config": hf_endpoint_url(
            "/google/vit-large-patch16-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-large-patch16-384/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-large-patch16-384/resolve/main/pytorch_model.bin"
        ),
    },
    "vit-large-patch32-384": {
        "config": hf_endpoint_url(
            "/google/vit-large-patch32-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/vit-large-patch32-384/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/google/vit-large-patch32-384/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.vit.modeling
import unitorch.cli.models.vit.processing
from unitorch.cli.models.vit.modeling import ViTForImageClassification
from unitorch.cli.models.vit.processing import ViTProcessor
