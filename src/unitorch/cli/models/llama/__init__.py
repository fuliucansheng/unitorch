# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_llama_infos = {
    # llama
    "llama-7b": {
        "config": hf_endpoint_url("/huggyllama/llama-7b/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/huggyllama/llama-7b/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/huggyllama/llama-7b/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/huggyllama/llama-7b/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/huggyllama/llama-7b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
            )
            for i in range(1, 3)
        ],
    },
    "llama-13b": {
        "config": hf_endpoint_url("/huggyllama/llama-13b/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/huggyllama/llama-13b/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/huggyllama/llama-13b/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/huggyllama/llama-13b/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/huggyllama/llama-13b/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            )
            for i in range(1, 4)
        ],
    },
    "llama2-7b": {
        "config": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-hf/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-hf/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-hf/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-hf/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/NousResearch/Llama-2-7b-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            )
            for i in range(1, 4)
        ],
    },
    "llama2-chat-7b": {
        "config": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-chat-hf/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/NousResearch/Llama-2-7b-chat-hf/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/NousResearch/Llama-2-7b-chat-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            )
            for i in range(1, 4)
        ],
    },
    "llama2-13b": {
        "config": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-hf/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-hf/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-hf/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-hf/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/NousResearch/Llama-2-13b-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00006.bin"
            )
            for i in range(1, 7)
        ],
    },
    "llama2-chat-13b": {
        "config": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-chat-hf/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-chat-hf/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-chat-hf/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/NousResearch/Llama-2-13b-chat-hf/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/NousResearch/Llama-2-13b-chat-hf/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00003.bin"
            )
            for i in range(1, 4)
        ],
    },
    "llama3-8b": {
        "config": hf_endpoint_url("/unsloth/llama-3-8b/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url("/unsloth/llama-3-8b/resolve/main/tokenizer.json"),
        "tokenizer_config": hf_endpoint_url(
            "/unsloth/llama-3-8b/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/unsloth/llama-3-8b/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/unsloth/llama-3-8b/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
            )
            for i in range(1, 5)
        ],
    },
    "llama3-13b": {
        "config": hf_endpoint_url("/Replete-AI/Llama-3-13B/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/Replete-AI/Llama-3-13B/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Replete-AI/Llama-3-13B/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/Replete-AI/Llama-3-13B/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/Replete-AI/Llama-3-13B/resolve/main/model-{str(i).rjust(5, '0')}-of-00003.safetensors"
            )
            for i in range(1, 4)
        ],
    },
    "llama3-8b-instruct": {
        "config": hf_endpoint_url(
            "/unsloth/llama-3-8b-Instruct/resolve/main/config.json"
        ),
        "tokenizer": hf_endpoint_url(
            "/unsloth/llama-3-8b-Instruct/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/unsloth/llama-3-8b-Instruct/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/unsloth/llama-3-8b-Instruct/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                f"/unsloth/llama-3-8b-Instruct/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00004.bin"
            )
            for i in range(1, 5)
        ],
    },
}

pretrained_llama_extensions_infos = {}

import unitorch.cli.models.llama.modeling
import unitorch.cli.models.llama.processing
