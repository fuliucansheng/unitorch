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
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen3-4B-Thinking-2507/resolve/main/model-{str(i).rjust(5, '0')}-of-00003.safetensors"
            )
            for i in range(1, 4)
        ],
    },
    "qwen3-14b": {
        "config": hf_endpoint_url("/Qwen/Qwen3-14B/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url("/Qwen/Qwen3-14B/resolve/main/tokenizer.json"),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-14B/resolve/main/tokenizer_config.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen3-14B/resolve/main/model-{str(i).rjust(5, '0')}-of-00008.safetensors"
            )
            for i in range(1, 9)
        ],
    },
    "qwen3-32b": {
        "config": hf_endpoint_url("/Qwen/Qwen3-32B/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url("/Qwen/Qwen3-32B/resolve/main/tokenizer.json"),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-32B/resolve/main/tokenizer_config.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen3-32B/resolve/main/model-{str(i).rjust(5, '0')}-of-00017.safetensors"
            )
            for i in range(1, 18)
        ],
    },
    "qwen3-vl-2b-instruct": {
        "config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/preprocessor_config.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json"
        ),
        "weight": hf_endpoint_url(
            f"/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors"
        ),
    },
    "qwen3-vl-8b-instruct": {
        "config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/tokenizer.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/preprocessor_config.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/chat_template.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen3-VL-8B-Instruct/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
            )
            for i in range(1, 5)
        ],
    },
    "qwen3-vl-8b-thinking": {
        "config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/tokenizer.json"
        ),
        "vision_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/preprocessor_config.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/chat_template.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Qwen/Qwen3-VL-8B-Thinking/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
            )
            for i in range(1, 5)
        ],
    },
}

pretrained_qwen_extensions_infos = {}

import unitorch.cli.models.qwen.modeling
import unitorch.cli.models.qwen.modeling_vl
import unitorch.cli.models.qwen.processing
import unitorch.cli.models.qwen.processing_vl
