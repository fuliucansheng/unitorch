# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_clip_infos = {
    "clip-vit-base-patch16": {
        "config": hf_endpoint_url(
            "/openai/clip-vit-base-patch16/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/openai/clip-vit-base-patch16/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/openai/clip-vit-base-patch16/resolve/main/vocab.json"
        ),
        "merge": hf_endpoint_url(
            "/openai/clip-vit-base-patch16/resolve/main/merges.txt"
        ),
        "weight": hf_endpoint_url(
            "/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin"
        ),
    },
    "clip-vit-base-patch32": {
        "config": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/vocab.json"
        ),
        "merge": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/merges.txt"
        ),
        "weight": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin"
        ),
    },
    "clip-vit-large-patch14": {
        "config": hf_endpoint_url(
            "/openai/clip-vit-large-patch14/resolve/main/config.json"
        ),
        "vision_config": hf_endpoint_url(
            "/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/openai/clip-vit-large-patch14/resolve/main/vocab.json"
        ),
        "merge": hf_endpoint_url(
            "/openai/clip-vit-large-patch14/resolve/main/merges.txt"
        ),
        "weight": hf_endpoint_url(
            "/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin"
        ),
    },
    # https://gist.github.com/rwightman/c79fd0241ed3c860e898114931c07990
    "clip-vit-huge-patch14-hpsv2": {
        "config": hf_endpoint_url("/adams-story/HPSv2-hf/resolve/main/config.json"),
        "vision_config": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json"
        ),
        "vocab": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/vocab.json"
        ),
        "merge": hf_endpoint_url(
            "/openai/clip-vit-base-patch32/resolve/main/merges.txt"
        ),
        "weight": hf_endpoint_url(
            "/adams-story/HPSv2-hf/resolve/main/pytorch_model.bin"
        ),
    },
}

pretrained_clip_extensions_infos = {}

import unitorch.cli.models.clip.modeling
import unitorch.cli.models.clip.processing
