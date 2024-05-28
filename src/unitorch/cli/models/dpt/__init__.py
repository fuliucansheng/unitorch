# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_dpt_infos = {
    "default-dpt": {
        "config": "https://huggingface.co/Intel/dpt-large/resolve/main/config.json",
        "vision_config": "https://huggingface.co/Intel/dpt-large/resolve/main/preprocessor_config.json",
    },
    "dpt-large": {
        "config": "https://huggingface.co/Intel/dpt-large/resolve/main/config.json",
        "vision_config": "https://huggingface.co/Intel/dpt-large/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/Intel/dpt-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.dpt.modeling
import unitorch.cli.models.dpt.processing
from unitorch.cli.models.dpt.modeling import DPTForDepthEstimation
from unitorch.cli.models.dpt.processing import DPTProcessor
