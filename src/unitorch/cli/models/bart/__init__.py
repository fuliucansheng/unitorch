# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_bart_infos = {
    "default-bart": {
        "config": "https://huggingface.co/facebook/bart-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "merge": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
    },
    "bart-base": {
        "config": "https://huggingface.co/facebook/bart-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "merge": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
        "weight": "https://huggingface.co/facebook/bart-base/resolve/main/pytorch_model.bin",
    },
    "bart-large": {
        "config": "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json",
        "merge": "https://huggingface.co/facebook/bart-large/resolve/main/merges.txt",
        "weight": "https://huggingface.co/facebook/bart-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.bart.modeling
import unitorch.cli.models.bart.processing
from unitorch.cli.models.bart.modeling import BartForGeneration
from unitorch.cli.models.bart.processing import BartProcessor
