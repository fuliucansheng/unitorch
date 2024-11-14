# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_mt5_infos = {
    "mt5-base": {
        "config": hf_endpoint_url("/google/mt5-base/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/google/mt5-base/resolve/main/spiece.model"),
        "weight": hf_endpoint_url("/google/mt5-base/resolve/main/pytorch_model.bin"),
    },
    "mt5-small": {
        "config": hf_endpoint_url("/google/mt5-small/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/google/mt5-small/resolve/main/spiece.model"),
        "weight": hf_endpoint_url("/google/mt5-small/resolve/main/pytorch_model.bin"),
    },
    "mt5-large": {
        "config": hf_endpoint_url("/google/mt5-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/google/mt5-large/resolve/main/spiece.model"),
        "weight": hf_endpoint_url("/google/mt5-large/resolve/main/pytorch_model.bin"),
    },
}

import unitorch.cli.models.mt5.modeling
import unitorch.cli.models.mt5.processing
from unitorch.cli.models.mt5.modeling import MT5ForGeneration
from unitorch.cli.models.mt5.processing import MT5Processor
