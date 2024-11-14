# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_mask2former_infos = {
    "mask2former-swin-tiny-ade-semantic": {
        "config": hf_endpoint_url(
            "/facebook/mask2former-swin-tiny-ade-semantic/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/facebook/mask2former-swin-tiny-ade-semantic/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/mask2former-swin-tiny-ade-semantic/resolve/main/pytorch_model.bin"
        ),
    },
    "mask2former-swin-base-ade-semantic": {
        "config": hf_endpoint_url(
            "/facebook/mask2former-swin-base-ade-semantic/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/facebook/mask2former-swin-base-ade-semantic/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/mask2former-swin-base-ade-semantic/resolve/main/pytorch_model.bin"
        ),
    },
    "mask2former-swin-large-ade-semantic": {
        "config": hf_endpoint_url(
            "/facebook/mask2former-swin-large-ade-semantic/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/facebook/mask2former-swin-large-ade-semantic/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/mask2former-swin-large-ade-semantic/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.mask2former.modeling
import unitorch.cli.models.mask2former.processing
from unitorch.cli.models.mask2former.modeling import Mask2FormerForSegmentation
from unitorch.cli.models.mask2former.processing import Mask2FormerProcessor
