# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_t5_infos = {
    "default-t5": {
        "config": "https://huggingface.co/t5-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/t5-base/resolve/main/spiece.model",
    },
    "t5-base": {
        "config": "https://huggingface.co/t5-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "weight": "https://huggingface.co/t5-base/resolve/main/pytorch_model.bin",
    },
    "t5-small": {
        "config": "https://huggingface.co/t5-small/resolve/main/config.json",
        "vocab": "https://huggingface.co/t5-small/resolve/main/spiece.model",
        "weight": "https://huggingface.co/t5-small/resolve/main/pytorch_model.bin",
    },
    "t5-large": {
        "config": "https://huggingface.co/t5-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "weight": "https://huggingface.co/t5-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.t5.modeling
import unitorch.cli.models.t5.processing
from unitorch.cli.models.t5.modeling import T5ForGeneration
from unitorch.cli.models.t5.processing import T5Processor
