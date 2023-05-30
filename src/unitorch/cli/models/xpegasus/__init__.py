# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_xpegasus_infos = {
    "default-xpegasus": {
        "config": "https://huggingface.co/google/pegasus-x-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/pegasus-x-base/resolve/main/spiece.model",
    },
    "xpegasus-base": {
        "config": "https://huggingface.co/google/pegasus-x-base/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/pegasus-x-base/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/pegasus-x-base/resolve/main/pytorch_model.bin",
    },
    "xpegasus-large": {
        "config": "https://huggingface.co/google/pegasus-x-large/resolve/main/config.json",
        "vocab": "https://huggingface.co/google/pegasus-x-large/resolve/main/spiece.model",
        "weight": "https://huggingface.co/google/pegasus-x-large/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.xpegasus.modeling
import unitorch.cli.models.xpegasus.processing
from unitorch.cli.models.xpegasus.modeling import XPegasusForGeneration
from unitorch.cli.models.xpegasus.processing import XPegasusProcessor
