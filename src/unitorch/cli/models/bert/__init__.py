# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_bert_infos = {
    "bert-base-uncased": {
        "config": hf_endpoint_url("/bert-base-uncased/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/bert-base-uncased/resolve/main/vocab.txt"),
        "weight": hf_endpoint_url("/bert-base-uncased/resolve/main/pytorch_model.bin"),
    },
    "distilbert-base-uncased": {
        "config": hf_endpoint_url("/distilbert-base-uncased/resolve/main/config.json"),
        "vocab": hf_endpoint_url("/distilbert-base-uncased/resolve/main/vocab.txt"),
        "weight": hf_endpoint_url(
            "/distilbert-base-uncased/resolve/main/pytorch_model.bin"
        ),
    },
    "bert-base-multilingual-uncased": {
        "config": hf_endpoint_url(
            "/bert-base-multilingual-uncased/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/bert-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/bert-base-multilingual-uncased/resolve/main/pytorch_model.bin"
        ),
    },
    "bert-base-multilingual-cased": {
        "config": hf_endpoint_url(
            "/bert-base-multilingual-cased/resolve/main/config.json"
        ),
        "vocab": hf_endpoint_url(
            "/bert-base-multilingual-cased/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/bert-base-multilingual-cased/resolve/main/pytorch_model.bin"
        ),
    },
}

import unitorch.cli.models.bert.modeling
import unitorch.cli.models.bert.processing
from unitorch.cli.models.bert.modeling import BertForClassification
from unitorch.cli.models.bert.processing import BertProcessor
