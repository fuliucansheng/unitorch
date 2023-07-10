# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_minigpt4_infos = {
    # llama
    "default-minigpt4": {
        "config": "https://huggingface.co/huggyllama/llama-7b/resolve/main/config.json",
        "vocab": "https://huggingface.co/huggyllama/llama-7b/resolve/main/tokenizer.model",
    },
    "minigpt4-7b": {
        "blip2_config_path": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/config.json",
        "llama_config_path": "https://huggingface.co/huggyllama/llama-7b/resolve/main/config.json",
        "vocab": "https://huggingface.co/huggyllama/llama-7b/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/preprocessor_config.json",
        "weight": [
            "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/pytorch_model-00001-of-00006.bin",
            "https://huggingface.co/fuliucansheng/minigpt4/resolve/main/pytorch_model.vicuna7b.bin",
            "https://huggingface.co/huggyllama/llama-7b/resolve/main/pytorch_model-00001-of-00002.bin",
            "https://huggingface.co/huggyllama/llama-7b/resolve/main/pytorch_model-00002-of-00002.bin",
        ],
    },
    "minigpt4-13b": {
        "blip2_config_path": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/config.json",
        "llama_config_path": "https://huggingface.co/huggyllama/llama-13b/resolve/main/config.json",
        "vocab": "https://huggingface.co/huggyllama/llama-13b/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/preprocessor_config.json",
        "weight": [
            "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/pytorch_model-00001-of-00006.bin",
            "https://huggingface.co/fuliucansheng/minigpt4/resolve/main/pytorch_model.vicuna13b.bin",
            "https://huggingface.co/huggyllama/llama-13b/resolve/main/pytorch_model-00001-of-00003.bin",
            "https://huggingface.co/huggyllama/llama-13b/resolve/main/pytorch_model-00002-of-00003.bin",
            "https://huggingface.co/huggyllama/llama-13b/resolve/main/pytorch_model-00003-of-00003.bin",
        ],
    },
}

import unitorch.cli.models.minigpt4.modeling
import unitorch.cli.models.minigpt4.processing
from unitorch.cli.models.minigpt4.modeling import MiniGPT4ViTLlamaForGeneration
from unitorch.cli.models.minigpt4.processing import MiniGPT4ViTLlamaProcessor
