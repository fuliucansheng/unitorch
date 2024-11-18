# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_sam_infos = {
    "sam-vit-base": {
        "config": hf_endpoint_url("/facebook/sam-vit-base/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/sam-vit-base/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/sam-vit-base/resolve/main/pytorch_model.bin"
        ),
    },
    "sam-vit-large": {
        "config": hf_endpoint_url("/facebook/sam-vit-large/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/sam-vit-large/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/sam-vit-large/resolve/main/pytorch_model.bin"
        ),
    },
    "sam-vit-huge": {
        "config": hf_endpoint_url("/facebook/sam-vit-huge/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/facebook/sam-vit-huge/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/facebook/sam-vit-huge/resolve/main/pytorch_model.bin"
        ),
    },
}

pretrained_sam_extensions_infos = {}

import unitorch.cli.models.sam.modeling
import unitorch.cli.models.sam.processing
from unitorch.cli.models.sam.modeling import SamForSegmentation
from unitorch.cli.models.sam.processing import SamProcessor
