# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_dpt_infos = {
    "dpt-large": {
        "config": hf_endpoint_url("/Intel/dpt-large/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/Intel/dpt-large/resolve/main/preprocessor_config.json"
        ),
        "weight": hf_endpoint_url("/Intel/dpt-large/resolve/main/pytorch_model.bin"),
    },
}

import unitorch.cli.models.dpt.modeling
import unitorch.cli.models.dpt.processing
from unitorch.cli.models.dpt.modeling import DPTForDepthEstimation
from unitorch.cli.models.dpt.processing import DPTProcessor
