# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_llava_infos = {
    "default-llava-v1.6": {
        "config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/preprocessor_config.json",
    },
    "llava-v1.6-mistral-7b-hf": {
        "config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/preprocessor_config.json",
        "weight": [
            f"https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
            for i in range(1, 5)
        ],
    },
}

pretrained_llava_extensions_infos = {}

import unitorch.cli.models.llava.modeling
import unitorch.cli.models.llava.processing
from unitorch.cli.models.llava.modeling import (
    LlavaMistralClipForClassification,
    LlavaMistralClipForGeneration,
)
from unitorch.cli.models.llava.processing import LlavaMistralClipProcessor
