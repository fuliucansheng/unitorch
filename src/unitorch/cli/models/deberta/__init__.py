# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos

pretrained_deberta_infos = {
    "default-deberta": {
        "config": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
        "merge": "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
    },
    "deberta-base": {
        "config": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
        "merge": "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
        "weight": "https://huggingface.co/microsoft/deberta-base/resolve/main/pytorch_model.bin",
    },
    "deberta-large": {
        "config": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-large/resolve/main/vocab.json",
        "merge": "https://huggingface.co/microsoft/deberta-large/resolve/main/merges.txt",
        "weight": "https://huggingface.co/microsoft/deberta-large/resolve/main/pytorch_model.bin",
    },
    "deberta-large-mnli": {
        "config": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/vocab.json",
        "merge": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/merges.txt",
        "weight": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/pytorch_model.bin",
    },
    "deberta-xlarge": {
        "config": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/vocab.json",
        "merge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/merges.txt",
        "weight": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/pytorch_model.bin",
    },
}

pretrained_deberta_v2_infos = {
    "default-deberta-v2": {
        "config": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
    },
    "deberta-v2-xlarge": {
        "config": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "weight": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/pytorch_model.bin",
    },
    "deberta-v2-xlarge": {
        "config": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "weight": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/pytorch_model.bin",
    },
    "deberta-v2-xxlarge": {
        "config": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "weight": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/pytorch_model.bin",
    },
    "deberta-v2-xlarge-mnli": {
        "config": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model",
        "weight": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/pytorch_model.bin",
    },
    "deberta-v2-xxlarge-mnli": {
        "config": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model",
        "weight": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.deberta.modeling
import unitorch.cli.models.deberta.processing
import unitorch.cli.models.deberta.modeling_v2
import unitorch.cli.models.deberta.processing_v2
from unitorch.cli.models.deberta.modeling import DebertaForClassification
from unitorch.cli.models.deberta.processing import DebertaProcessor
from unitorch.cli.models.deberta.modeling_v2 import DebertaV2ForClassification
from unitorch.cli.models.deberta.processing_v2 import DebertaV2Processor
