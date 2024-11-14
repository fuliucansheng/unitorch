# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_visualbert_infos = {
    "visualbert-vqa-coco-pre": {
        "config": hf_endpoint_url(
            "/uclanlp/visualbert-vqa-coco-pre/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url("/bert-base-uncased/resolve/main/vocab.txt"),
        "weight": hf_endpoint_url(
            "/uclanlp/visualbert-vqa-coco-pre/resolve/main/pytorch_model.bin"
        ),
    },
    "visualbert-nlvr2-coco-pre": {
        "config": hf_endpoint_url(
            "/uclanlp/visualbert-nlvr2-coco-pre/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/bert-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/uclanlp/visualbert-nlvr2-coco-pre/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.visualbert.modeling
import unitorch.cli.models.visualbert.processing
from unitorch.cli.models.visualbert.modeling import (
    VisualBertForClassification,
    VisualBertForPretrain,
)
from unitorch.cli.models.visualbert.processing import VisualBertProcessor
