# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_llama_infos = {
    # llama
    "default-llama": {
        "config": "https://huggingface.co/huggyllama/llama-7b/resolve/main/config.json",
        "vocab": "https://huggingface.co/huggyllama/llama-7b/resolve/main/tokenizer.model",
    },
    "llama-7b-hf": {
        "config": "https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00033.bin"
            for i in range(1, 34)
        ],
    },
    "llama-7b": {
        "config": "https://huggingface.co/huggyllama/llama-7b/resolve/main/config.json",
        "vocab": "https://huggingface.co/huggyllama/llama-7b/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/huggyllama/llama-7b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
            for i in range(1, 3)
        ],
    },
}

import unitorch.cli.models.llama.modeling
import unitorch.cli.models.llama.processing
from unitorch.cli.models.llama.modeling import (
    LlamaForClassification,
    LlamaForPretrain,
    LlamaForGeneration,
)
from unitorch.cli.models.llama.processing import LlamaProcessor