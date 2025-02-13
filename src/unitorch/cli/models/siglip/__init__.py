# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_siglip_infos = {
    "siglip-base-patch16-224": {
        "config": hf_endpoint_url(
            "/google/siglip-base-patch16-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/siglip-base-patch16-224/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/siglip-base-patch16-224/resolve/main/spiece.model"
        ),
        "weight": hf_endpoint_url(
            "/google/siglip-base-patch16-224/resolve/main/model.safetensors"
        ),
    },
    "siglip-so400m-patch14-384": {
        "config": hf_endpoint_url(
            "/google/siglip-so400m-patch14-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/siglip-so400m-patch14-384/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/siglip-so400m-patch14-384/resolve/main/spiece.model"
        ),
        "weight": hf_endpoint_url(
            "/google/siglip-so400m-patch14-384/resolve/main/model.safetensors"
        ),
    },
}

pretrained_siglip_extensions_infos = {}

import unitorch.cli.models.siglip.modeling
import unitorch.cli.models.siglip.processing
from unitorch.cli.models.siglip.modeling import (
    SiglipForPretrain,
    SiglipForClassification,
    SiglipForTextClassification,
    SiglipForImageClassification,
)
from unitorch.cli.models.siglip.processing import SiglipProcessor
