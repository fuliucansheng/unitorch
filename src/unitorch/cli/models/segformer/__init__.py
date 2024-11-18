# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_segformer_infos = {
    "segformer-b2-human-parse-24": {
        "config": hf_endpoint_url(
            "/yolo12138/segformer-b2-human-parse-24/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/yolo12138/segformer-b2-human-parse-24/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/yolo12138/segformer-b2-human-parse-24/resolve/main/model.safetensors"
        ),
    },
    "segformer-b5-finetuned-human-parsing": {
        "config": hf_endpoint_url(
            "/matei-dorian/segformer-b5-finetuned-human-parsing/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/matei-dorian/segformer-b5-finetuned-human-parsing/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/matei-dorian/segformer-b5-finetuned-human-parsing/resolve/main/model.safetensors"
        ),
    },
    "segformer-face-parsing": {
        "config": hf_endpoint_url(
            "/jonathandinu/face-parsing/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/jonathandinu/face-parsing/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url(
            "/jonathandinu/face-parsing/resolve/main/model.safetensors"
        ),
    },
}

import unitorch.cli.models.segformer.modeling
import unitorch.cli.models.segformer.processing
from unitorch.cli.models.segformer.modeling import SegformerForSegmentation
from unitorch.cli.models.segformer.processing import SegformerProcessor
