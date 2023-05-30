# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_beit_infos = {
    "default-beit": {
        "config": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/preprocessor_config.json",
    },
    "beit-base-patch16-224-pt22k-ft22k": {
        "config": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k/resolve/main/pytorch_model.bin",
    },
    "beit-base-patch16-224": {
        "config": "https://huggingface.co/microsoft/beit-base-patch16-224/resolve/main/config.json",
        "vision_config": "https://huggingface.co/microsoft/beit-base-patch16-224/resolve/main/preprocessor_config.json",
        "weight": "https://huggingface.co/microsoft/beit-base-patch16-224/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.beit.modeling
import unitorch.cli.models.beit.processing
from unitorch.cli.models.beit.modeling import BeitForImageClassification
from unitorch.cli.models.beit.processing import BeitProcessor
