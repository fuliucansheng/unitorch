# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_xprophetnet_infos = {
    "default-xprophetnet": {
        "config": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/prophetnet.tokenizer",
    },
    "xprophetnet-large-wiki100-cased": {
        "config": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json",
        "vocab": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/prophetnet.tokenizer",
        "weight": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.xprophetnet.modeling
import unitorch.cli.models.xprophetnet.processing
from unitorch.cli.models.xprophetnet.modeling import XProphetNetForGeneration
from unitorch.cli.models.xprophetnet.processing import XProphetNetProcessor
