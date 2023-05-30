# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_mbart_infos = {
    "default-mbart": {
        "config": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json",
        "vocab": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentence.bpe.model",
    },
    "mbart-large-cc25": {
        "config": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json",
        "vocab": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentence.bpe.model",
        "weight": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.mbart.modeling
import unitorch.cli.models.mbart.processing
from unitorch.cli.models.mbart.modeling import MBartForGeneration
from unitorch.cli.models.mbart.processing import MBartProcessor
