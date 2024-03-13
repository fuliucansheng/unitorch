# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_mistral_infos = {
    # mistral
    "default-mistral": {
        "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
        "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
    },
    "mistral-7b-instruct-v0.1": {
        "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
        "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
            for i in range(1, 3)
        ],
    },
}

import unitorch.cli.models.mistral.modeling
import unitorch.cli.models.mistral.processing
from unitorch.cli.models.mistral.modeling import (
    MistralForGeneration,
)
from unitorch.cli.models.mistral.processing import MistralProcessor
