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
}

import unitorch.cli.models.bert.modeling
import unitorch.cli.models.bert.processing
