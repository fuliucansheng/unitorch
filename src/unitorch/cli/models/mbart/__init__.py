# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_mbart_infos = {
    "mbart-large-cc25": {
        "config": hf_endpoint_url(
            "/facebook/mbart-large-cc25/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/facebook/mbart-large-cc25/resolve/main/sentence.bpe.model"
        ),
        "weight": hf_endpoint_url(
            "/facebook/mbart-large-cc25/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.mbart.modeling
import unitorch.cli.models.mbart.processing
