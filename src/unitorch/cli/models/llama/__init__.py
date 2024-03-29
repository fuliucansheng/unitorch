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
    "llama-13b-hf": {
        "config": "https://huggingface.co/decapoda-research/llama-13b-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/decapoda-research/llama-13b-hf/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/decapoda-research/llama-13b-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00041.bin"
            for i in range(0, 42)
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
    "llama-13b": {
        "config": "https://huggingface.co/huggyllama/llama-13b/resolve/main/config.json",
        "vocab": "https://huggingface.co/huggyllama/llama-13b/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/huggyllama/llama-13b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            for i in range(1, 4)
        ],
    },
    "llama2-7b": {
        "config": "https://huggingface.co/NousResearch/Llama-2-7b-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/NousResearch/Llama-2-7b-hf/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/NousResearch/Llama-2-7b-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            for i in range(1, 4)
        ],
    },
    "llama2-chat-7b": {
        "config": "https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            for i in range(1, 4)
        ],
    },
    "llama2-13b": {
        "config": "https://huggingface.co/NousResearch/Llama-2-13b-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/NousResearch/Llama-2-13b-hf/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/NousResearch/Llama-2-13b-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00006.bin"
            for i in range(1, 7)
        ],
    },
    "llama2-chat-13b": {
        "config": "https://huggingface.co/NousResearch/Llama-2-13b-chat-hf/resolve/main/config.json",
        "vocab": "https://huggingface.co/NousResearch/Llama-2-13b-chat-hf/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/NousResearch/Llama-2-13b-chat-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            for i in range(1, 4)
        ],
    },
    "open-llama-3b": {
        "config": "https://huggingface.co/openlm-research/open_llama_3b/resolve/main/config.json",
        "vocab": "https://huggingface.co/openlm-research/open_llama_3b/resolve/main/tokenizer.model",
        "weight": "https://huggingface.co/openlm-research/open_llama_3b/resolve/main/pytorch_model.bin",
    },
    "open-llama-7b": {
        "config": "https://huggingface.co/openlm-research/open_llama_7b/resolve/main/config.json",
        "vocab": "https://huggingface.co/openlm-research/open_llama_7b/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/openlm-research/open_llama_7b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
            for i in range(1, 3)
        ],
    },
    "open-llama-13b": {
        "config": "https://huggingface.co/openlm-research/open_llama_13b/resolve/main/config.json",
        "vocab": "https://huggingface.co/openlm-research/open_llama_13b/resolve/main/tokenizer.model",
        "weight": [
            f"https://huggingface.co/openlm-research/open_llama_13b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            for i in range(1, 4)
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
