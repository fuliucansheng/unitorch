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
}

pretrained_siglip_extensions_infos = {}

import unitorch.cli.models.siglip.modeling
import unitorch.cli.models.siglip.processing
