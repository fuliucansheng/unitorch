# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_llava_infos = {
    "llava-v1.6-mistral-7b-hf": {
        "config": hf_endpoint_url(
            "/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/tokenizer.model"
        ),
        "vision_config": hf_endpoint_url(
            "/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/preprocessor_config.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
            )
            for i in range(1, 5)
        ],
    },
}

pretrained_llava_extensions_infos = {}

import unitorch.cli.models.llava.modeling
import unitorch.cli.models.llava.processing
from unitorch.cli.models.llava.modeling import (
    LlavaMistralClipForClassification,
    LlavaMistralClipForGeneration,
)
from unitorch.cli.models.llava.processing import LlavaMistralClipProcessor
