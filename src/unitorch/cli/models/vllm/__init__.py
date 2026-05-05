# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

pretrained_vllm_infos = {
    "qwen3-4b-thinking": {
        "hf_pretrained_name": "Qwen/Qwen3-4B-Thinking-2507",
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-4B-Thinking-2507/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-4B-Thinking-2507/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-4B-Thinking-2507/resolve/main/chat_template.json"
        ),
    },
    "qwen3-8b": {
        "hf_pretrained_name": "Qwen/Qwen3-8B",
        "tokenizer": hf_endpoint_url("/Qwen/Qwen3-8B/resolve/main/tokenizer.json"),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-8B/resolve/main/tokenizer_config.json"
        ),
    },
    "qwen3-14b": {
        "hf_pretrained_name": "Qwen/Qwen3-14B",
        "tokenizer": hf_endpoint_url("/Qwen/Qwen3-14B/resolve/main/tokenizer.json"),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-14B/resolve/main/tokenizer_config.json"
        ),
    },
    "qwen3-32b": {
        "hf_pretrained_name": "Qwen/Qwen3-32B",
        "tokenizer": hf_endpoint_url("/Qwen/Qwen3-32B/resolve/main/tokenizer.json"),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-32B/resolve/main/tokenizer_config.json"
        ),
    },
    "qwen3-vl-2b-instruct": {
        "hf_pretrained_name": "Qwen/Qwen3-VL-2B-Instruct",
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json"
        ),
    },
    "qwen3-vl-8b-instruct": {
        "hf_pretrained_name": "Qwen/Qwen3-VL-8B-Instruct",
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Instruct/resolve/main/chat_template.json"
        ),
    },
    "qwen3-vl-8b-thinking": {
        "hf_pretrained_name": "Qwen/Qwen3-VL-8B-Thinking",
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-8B-Thinking/resolve/main/chat_template.json"
        ),
    },
}

from unitorch.utils import is_vllm_available

if is_vllm_available():
    import unitorch.cli.models.vllm.modeling
    import unitorch.cli.models.vllm.modeling_vl
