# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_detr_infos = {
    "detr-resnet-50": {
        "config": hf_endpoint_url("/facebook/detr-resnet-50/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/detr-resnet-50/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/detr-resnet-50/resolve/main/pytorch_model.bin"
        ),
    },
    "detr-resnet-101": {
        "config": hf_endpoint_url("/facebook/detr-resnet-101/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/detr-resnet-101/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/detr-resnet-101/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.detr.modeling
import unitorch.cli.models.detr.processing
from unitorch.cli.models.detr.modeling import (
    DetrForDetection,
)
from unitorch.cli.models.detr.processing import DetrProcessor
