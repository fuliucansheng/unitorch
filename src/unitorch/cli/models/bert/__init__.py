# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos

pretrained_bert_infos = {
    "default-bert": {
        "config": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    },
    "bert-base-uncased": {
        "config": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin",
    },
    "distilbert-base-uncased": {
        "config": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
        "vocab": "https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin",
    },
    "bert-base-multilingual-uncased": {
        "config": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/pytorch_model.bin",
    },
    "bert-base-multilingual-cased": {
        "config": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
        "vocab": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.bert.modeling
import unitorch.cli.models.bert.processing
from unitorch.cli.models.bert.modeling import BertForClassification
from unitorch.cli.models.bert.processing import BertProcessor
