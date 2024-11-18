# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_chinese_clip_infos = {
    "chinese-clip-vit-base-patch16": {
        "config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/pytorch_model.bin"
        ),
    },
    "chinese-clip-vit-large-patch14": {
        "config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-large-patch14/resolve/main/pytorch_model.bin"
        ),
    },
    "chinese-clip-vit-huge-patch14": {
        "config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14/resolve/main/pytorch_model.bin"
        ),
    },
    "chinese-clip-vit-huge-patch14-336px": {
        "config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/vocab.txt"
        ),
        "weight": hf_endpoint_url(
            "/OFA-Sys/chinese-clip-vit-huge-patch14-336px/resolve/main/pytorch_model.bin"
        ),
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
