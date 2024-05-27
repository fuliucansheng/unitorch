# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

# pretrained infos
pretrained_chinese_clip_infos = {
    "default-chinese-clip": {
        "config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json",
        "vision_config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/vocab.txt",
    },
    "chinese-clip-vit-base-patch16": {
        "config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json",
        "vision_config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/pytorch_model.bin",
    },
    "chinese-clip-vit-large-patch14": {
        "config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/config.json",
        "vision_config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/pytorch_model.bin",
    },
    "chinese-clip-vit-huge-patch14": {
        "config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/config.json",
        "vision_config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/pytorch_model.bin",
    },
    "chinese-clip-vit-huge-patch14-336px": {
        "config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/config.json",
        "vision_config": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/preprocessor_config.json",
        "vocab": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/vocab.txt",
        "weight": "https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/pytorch_model.bin",
    },
}

import unitorch.cli.models.chinese_clip.modeling
import unitorch.cli.models.chinese_clip.processing
from unitorch.cli.models.chinese_clip.modeling import (
    ChineseClipForPretrain,
    ChineseClipForClassification,
    ChineseClipForTextClassification,
    ChineseClipForImageClassification,
)
from unitorch.cli.models.chinese_clip.processing import ChineseClipProcessor
