# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_grounding_dino_infos = {
    "grounding-dino-tiny": {
        "config": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-tiny/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-tiny/resolve/main/vocab.txt"
        ),
        "vision_config": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-tiny/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-tiny/resolve/main/model.safetensors"
        ),
    },
    "grounding-dino-base": {
        "config": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-base/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-base/resolve/main/vocab.txt"
        ),
        "vision_config": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-base/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/IDEA-Research/grounding-dino-base/resolve/main/model.safetensors"
        ),
    },
}

import unitorch.cli.models.grounding_dino.modeling
import unitorch.cli.models.grounding_dino.processing
from unitorch.cli.models.grounding_dino.modeling import (
    GroundingDinoForDetection,
)
from unitorch.cli.models.grounding_dino.processing import GroundingDinoProcessor
