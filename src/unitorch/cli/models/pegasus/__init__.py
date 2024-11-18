# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_pegasus_infos = {
    "pegasus-cnn_dailymail": {
        "config": hf_endpoint_url(
            "/google/pegasus-cnn_dailymail/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/google/pegasus-cnn_dailymail/resolve/main/spiece.model"
        ),
        "weight": hf_endpoint_url(
            "/google/pegasus-cnn_dailymail/resolve/main/pytorch_model.bin"
        ),
    },
    "pegasus-xsum": {
        "config": hf_endpoint_url("/google/pegasus-xsum/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/google/pegasus-xsum/resolve/main/spiece.model"),
        "weight": hf_endpoint_url(
            "/google/pegasus-xsum/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.pegasus.modeling
import unitorch.cli.models.pegasus.processing
from unitorch.cli.models.pegasus.modeling import PegasusForGeneration
from unitorch.cli.models.pegasus.processing import PegasusProcessor
