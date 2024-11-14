# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_dinov2_infos = {
    "dinov2-base": {
        "config": hf_endpoint_url("/facebook/dinov2-base/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/dinov2-base/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/dinov2-base/resolve/main/pytorch_model.bin"
        ),
    },
    "dinov2-small": {
        "config": hf_endpoint_url("/facebook/dinov2-small/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/dinov2-small/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/dinov2-small/resolve/main/pytorch_model.bin"
        ),
    },
    "dinov2-large": {
        "config": hf_endpoint_url("/facebook/dinov2-large/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/dinov2-large/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/dinov2-large/resolve/main/pytorch_model.bin"
        ),
    },
    "dinov2-giant": {
        "config": hf_endpoint_url("/facebook/dinov2-giant/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/dinov2-giant/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/dinov2-giant/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.dinov2.modeling
import unitorch.cli.models.dinov2.processing
from unitorch.cli.models.dinov2.modeling import DinoV2ForImageClassification
from unitorch.cli.models.dinov2.processing import DinoV2Processor
