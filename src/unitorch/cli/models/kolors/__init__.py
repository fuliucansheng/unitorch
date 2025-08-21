# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_kolors_infos = {
    "kolors-mps-overall": {
        "config": hf_endpoint_url(
            "/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/vocab.json"
        ),
        "merge": hf_endpoint_url(
            "/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/merges.txt"
        ),
        "weight": hf_endpoint_url(
            "/datasets/fuliucansheng/hubfiles/resolve/main/kolors/pytorch_model.mps.overall.bin"
        ),
    },
}

pretrained_kolors_extensions_infos = {}

import unitorch.cli.models.kolors.modeling
import unitorch.cli.models.kolors.processing
