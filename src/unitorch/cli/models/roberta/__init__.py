# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_roberta_infos = {
    "roberta-base": {
        "config": hf_endpoint_url("/roberta-base/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/roberta-base/resolve/main/vocab.json"),
        "merge": hf_endpoint_url("/roberta-base/resolve/main/merges.txt"),
        "weight": hf_endpoint_url("/roberta-base/resolve/main/pytorch_model.bin"),
    },
    "roberta-large": {
        "config": hf_endpoint_url("/roberta-large/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/roberta-large/resolve/main/vocab.json"),
        "merge": hf_endpoint_url("/roberta-large/resolve/main/merges.txt"),
        "weight": hf_endpoint_url("/roberta-large/resolve/main/pytorch_model.bin"),
    },
}

import unitorch.cli.models.roberta.modeling
import unitorch.cli.models.roberta.processing
from unitorch.cli.models.roberta.modeling import RobertaForClassification
from unitorch.cli.models.roberta.processing import RobertaProcessor
