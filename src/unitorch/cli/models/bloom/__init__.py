# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_bloom_infos = {
    # bloom
    "default-bloom": {
        "config": "https://huggingface.co/bigscience/bloom-560m/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloom-560m/resolve/main/tokenizer.json",
    },
    "bloom-560m": {
        "config": "https://huggingface.co/bigscience/bloom-560m/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloom-560m/resolve/main/tokenizer.json",
        "weight": "https://huggingface.co/bigscience/bloom-560m/resolve/main/pytorch_model.bin",
    },
    "bloom-3b": {
        "config": "https://huggingface.co/bigscience/bloom-3b/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloom-3b/resolve/main/tokenizer.json",
        "weight": "https://huggingface.co/bigscience/bloom-3b/resolve/main/pytorch_model.bin",
    },
    "bloom-7b1": {
        "config": "https://huggingface.co/bigscience/bloom-7b1/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloom-7b1/resolve/main/tokenizer.json",
        "weight": [
            f"https://huggingface.co/bigscience/bloom-7b1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
            for i in range(1, 3)
        ],
    },
    "bloomz-560m": {
        "config": "https://huggingface.co/bigscience/bloomz-560m/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloomz-560m/resolve/main/tokenizer.json",
        "weight": "https://huggingface.co/bigscience/bloomz-560m/resolve/main/pytorch_model.bin",
    },
    "bloomz-3b": {
        "config": "https://huggingface.co/bigscience/bloomz-3b/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloomz-3b/resolve/main/tokenizer.json",
        "weight": "https://huggingface.co/bigscience/bloomz-3b/resolve/main/pytorch_model.bin",
    },
    "bloomz-7b1": {
        "config": "https://huggingface.co/bigscience/bloomz-7b1/resolve/main/config.json",
        "tokenizer": "https://huggingface.co/bigscience/bloomz-7b1/resolve/main/tokenizer.json",
        "weight": "https://huggingface.co/bigscience/bloomz-7b1/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.bloom.modeling
import unitorch.cli.models.bloom.processing
from unitorch.cli.models.bloom.modeling import (
    BloomForClassification,
    BloomForPretrain,
    BloomForGeneration,
)
from unitorch.cli.models.bloom.processing import BloomProcessor
