# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_clip_infos = {
    "default-clip": {
        "config": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/config.json",
        "vision_config": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/vocab.json",
        "merge": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/merges.txt",
    },
    "clip-vit-base-patch16": {
        "config": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/config.json",
        "vision_config": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/vocab.json",
        "merge": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/merges.txt",
        "weight": "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin",
    },
    "clip-vit-base-patch32": {
        "config": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
        "vision_config": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
        "merge": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
        "weight": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin",
    },
    "clip-vit-large-patch14": {
        "config": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json",
        "vision_config": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json",
        "merge": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt",
        "weight": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.clip.modeling
import unitorch.cli.models.clip.processing
from unitorch.cli.models.clip.modeling import (
    ClipForPretrain,
    ClipForClassification,
    ClipForTextClassification,
    ClipForImageClassification,
)
from unitorch.cli.models.clip.processing import ClipProcessor
