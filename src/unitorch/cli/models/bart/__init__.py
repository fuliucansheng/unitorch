# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_bart_infos = {
    "bart-base": {
        "config": hf_endpoint_url("/facebook/bart-base/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/facebook/bart-base/resolve/main/vocab.json"),
        "merge": hf_endpoint_url("/facebook/bart-base/resolve/main/merges.txt"),
        "weight": hf_endpoint_url("/facebook/bart-base/resolve/main/pytorch_model.bin"),
    },
    "bart-large": {
        "config": hf_endpoint_url("/facebook/bart-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/facebook/bart-large/resolve/main/vocab.json"),
        "merge": hf_endpoint_url("/facebook/bart-large/resolve/main/merges.txt"),
        "weight": hf_endpoint_url(
            "/facebook/bart-large/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.bart.modeling
import unitorch.cli.models.bart.processing
from unitorch.cli.models.bart.modeling import BartForGeneration
from unitorch.cli.models.bart.processing import BartProcessor
