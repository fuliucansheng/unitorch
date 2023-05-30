# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_mt5_infos = {
    "default-mt5": {
        "config": "https://huggingface.co/google/mt5-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/mt5-base/resolve/main/spiece.model",
    },
    "mt5-base": {
        "config": "https://huggingface.co/google/mt5-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/mt5-base/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/mt5-base/resolve/main/pytorch_model.bin",
    },
    "mt5-small": {
        "config": "https://huggingface.co/google/mt5-small/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/mt5-small/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/mt5-small/resolve/main/pytorch_model.bin",
    },
    "mt5-large": {
        "config": "https://huggingface.co/google/mt5-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/mt5-large/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/mt5-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.mt5.modeling
import unitorch.cli.models.mt5.processing
from unitorch.cli.models.mt5.modeling import MT5ForGeneration
from unitorch.cli.models.mt5.processing import MT5Processor
