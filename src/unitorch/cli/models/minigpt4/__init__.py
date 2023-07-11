# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_minigpt4_infos = {
    # llama
    "default-minigpt4": {
        "blip2_config_path": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/config.json",
        "llama_config_path": "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/config.json",
        "vocab": "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/preprocessor_config.json",
    },
    "minigpt4-7b": {
        "blip2_config_path": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/config.json",
        "llama_config_path": "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/config.json",
        "vocab": "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/preprocessor_config.json",
        "weight": [
            "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/pytorch_model-00001-of-00006.bin",
            "https://huggingface.co/fuliucansheng/minigpt4/resolve/main/pytorch_model.vicuna7b.bin",
            "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/pytorch_model-00001-of-00002.bin",
            "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA-7B/resolve/main/pytorch_model-00002-of-00002.bin",
        ],
    },
    "minigpt4-13b": {
        "blip2_config_path": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/config.json",
        "llama_config_path": "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA/resolve/main/config.json",
        "vocab": "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA/resolve/main/tokenizer.model",
        "vision_config": "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/preprocessor_config.json",
        "weight": [
            "https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/pytorch_model-00001-of-00006.bin",
            "https://huggingface.co/fuliucansheng/minigpt4/resolve/main/pytorch_model.vicuna13b.bin",
            "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA/resolve/main/pytorch_model-00001-of-00003.bin",
            "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA/resolve/main/pytorch_model-00002-of-00003.bin",
            "https://huggingface.co/wangrongsheng/MiniGPT-4-LLaMA/resolve/main/pytorch_model-00003-of-00003.bin",
        ],
    },
}

import unitorch.cli.models.minigpt4.modeling
import unitorch.cli.models.minigpt4.processing
from unitorch.cli.models.minigpt4.modeling import MiniGPT4Blip2LlamaForGeneration
from unitorch.cli.models.minigpt4.processing import MiniGPT4Blip2LlamaProcessor
