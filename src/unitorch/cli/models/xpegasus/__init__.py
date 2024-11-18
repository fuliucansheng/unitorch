# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_xpegasus_infos = {
    "xpegasus-base": {
        "config": hf_endpoint_url("/google/pegasus-x-base/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/google/pegasus-x-base/resolve/main/spiece.model"),
        "weight": hf_endpoint_url(
            "/google/pegasus-x-base/resolve/main/pytorch_model.bin"
        ),
    },
    "xpegasus-large": {
        "config": hf_endpoint_url("/google/pegasus-x-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/google/pegasus-x-large/resolve/main/spiece.model"),
        "weight": hf_endpoint_url(
            "/google/pegasus-x-large/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.xpegasus.modeling
import unitorch.cli.models.xpegasus.processing
from unitorch.cli.models.xpegasus.modeling import XPegasusForGeneration
from unitorch.cli.models.xpegasus.processing import XPegasusProcessor
