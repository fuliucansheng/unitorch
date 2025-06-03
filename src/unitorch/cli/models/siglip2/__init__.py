# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_siglip2_infos = {
    "siglip2-base-patch16-224": {
        "config": hf_endpoint_url(
            "/google/siglip2-base-patch16-224/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/siglip2-base-patch16-224/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/siglip2-base-patch16-224/resolve/main/tokenizer.model"
        ),
        "weight": hf_endpoint_url(
            "/google/siglip2-base-patch16-224/resolve/main/model.safetensors"
        ),
    },
    "siglip2-so400m-patch14-384": {
        "config": hf_endpoint_url(
            "/google/siglip2-so400m-patch14-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/siglip2-so400m-patch14-384/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/siglip2-so400m-patch14-384/resolve/main/tokenizer.model"
        ),
        "weight": hf_endpoint_url(
            "/google/siglip2-so400m-patch14-384/resolve/main/model.safetensors"
        ),
    },
    "siglip2-so400m-patch16-512": {
        "config": hf_endpoint_url(
            "/google/siglip2-so400m-patch16-512/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/siglip2-so400m-patch16-512/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/siglip2-so400m-patch16-512/resolve/main/tokenizer.model"
        ),
        "weight": hf_endpoint_url(
            "/google/siglip2-so400m-patch16-512/resolve/main/model.safetensors"
        ),
    },
    "siglip2-giant-opt-patch16-384": {
        "config": hf_endpoint_url(
            "/google/siglip2-giant-opt-patch16-384/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/google/siglip2-giant-opt-patch16-384/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/siglip2-giant-opt-patch16-384/resolve/main/tokenizer.model"
        ),
        "weight": [
            hf_endpoint_url(
                "/google/siglip2-giant-opt-patch16-384/resolve/main/model-00001-of-00002.safetensors"
            ),
            hf_endpoint_url(
                "/google/siglip2-giant-opt-patch16-384/resolve/main/model-00002-of-00002.safetensors"
            ),
        ],
    },
}

pretrained_siglip2_extensions_infos = {}

import unitorch.cli.models.siglip2.modeling
import unitorch.cli.models.siglip2.processing
from unitorch.cli.models.siglip2.modeling import (
    Siglip2ForPretrain,
    Siglip2ForClassification,
    Siglip2ForTextClassification,
    Siglip2ForImageClassification,
)
from unitorch.cli.models.siglip2.processing import Siglip2Processor
