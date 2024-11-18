# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_t5_infos = {
    "t5-base": {
        "config": hf_endpoint_url("/t5-base/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/t5-base/resolve/main/spiece.model"),
        "weight": hf_endpoint_url("/t5-base/resolve/main/pytorch_model.bin"),
    },
    "t5-small": {
        "config": hf_endpoint_url("/t5-small/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/t5-small/resolve/main/spiece.model"),
        "weight": hf_endpoint_url("/t5-small/resolve/main/pytorch_model.bin"),
    },
    "t5-large": {
        "config": hf_endpoint_url("/t5-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/t5-large/resolve/main/spiece.model"),
        "weight": hf_endpoint_url("/t5-large/resolve/main/pytorch_model.bin"),
    },
}

import unitorch.cli.models.t5.modeling
import unitorch.cli.models.t5.processing
from unitorch.cli.models.t5.modeling import T5ForGeneration
from unitorch.cli.models.t5.processing import T5Processor
