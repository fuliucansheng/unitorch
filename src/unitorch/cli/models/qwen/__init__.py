# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_qwen_infos = {
    "qwen3-4b-thinking": {
        "config": hf_endpoint_url(
            "/Qwen/Qwen3-4B-Thinking-2507/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-4B-Thinking-2507/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-4B-Thinking-2507/resolve/main/tokenizer_config.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen3-4B-Thinking-2507/resolve/main/model-{str(i).rjust(5, '0')}-of-00003.safetensors"
            )
            for i in range(1, 4)
        ],
    },
    "qwen2_5-vl-3b-instruct": {
        "config": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/tokenizer.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/preprocessor_config.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-{str(i).rjust(5, '0')}-of-00002.safetensors"
            )
            for i in range(1, 3)
        ],
    },
    "qwen2_5-vl-7b-instruct": {
        "config": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/tokenizer.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-{str(i).rjust(5, '0')}-of-00005.safetensors"
            )
            for i in range(1, 6)
        ],
    },
}

pretrained_qwen_extensions_infos = {}

import unitorch.cli.models.qwen.modeling
import unitorch.cli.models.qwen.modeling_vl
import unitorch.cli.models.qwen.processing
import unitorch.cli.models.qwen.processing_vl
from unitorch.cli.models.qwen.modeling import (
    QWen3ForGeneration,
)
from unitorch.cli.models.qwen.modeling_vl import QWen2_5VLForGeneration
from unitorch.cli.models.qwen.processing import QWenProcessor
from unitorch.cli.models.qwen.processing_vl import QWenVLProcessor
