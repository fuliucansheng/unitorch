# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_pegasus_infos = {
    "default-pegasus": {
        "config": "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model",
    },
    "pegasus-cnn_dailymail": {
        "config": "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/pytorch_model.bin",
    },
    "pegasus-xsum": {
        "config": "https://huggingface.co/google/pegasus-xsum/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/pegasus-xsum/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.pegasus.modeling
import unitorch.cli.models.pegasus.processing
from unitorch.cli.models.pegasus.modeling import PegasusForGeneration
from unitorch.cli.models.pegasus.processing import PegasusProcessor
