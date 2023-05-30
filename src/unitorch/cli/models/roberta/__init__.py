# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos

pretrained_roberta_infos = {
    "default-roberta": {
        "config": "https://huggingface.co/roberta-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "merge": "https://huggingface.co/roberta-base/resolve/main/merges.txt",
    },
    "roberta-base": {
        "config": "https://huggingface.co/roberta-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "merge": "https://huggingface.co/roberta-base/resolve/main/merges.txt",
        "weight": "https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin",
    },
    "roberta-large": {
        "config": "https://huggingface.co/roberta-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/roberta-large/resolve/main/vocab.json",
        "merge": "https://huggingface.co/roberta-large/resolve/main/merges.txt",
        "weight": "https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.roberta.modeling
import unitorch.cli.models.roberta.processing
from unitorch.cli.models.roberta.modeling import RobertaForClassification
from unitorch.cli.models.roberta.processing import RobertaProcessor
