# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url

# pretrained infos
pretrained_bloom_infos = {
    # bloom
    "bloom-560m": {
        "config": hf_endpoint_url("/bigscience/bloom-560m/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/bigscience/bloom-560m/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/bigscience/bloom-560m/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/bigscience/bloom-560m/resolve/main/special_tokens_map.json"
        ),
        "weight": hf_endpoint_url(
            "/bigscience/bloom-560m/resolve/main/pytorch_model.bin"
        ),
    },
    "bloom-3b": {
        "config": hf_endpoint_url("/bigscience/bloom-3b/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/bigscience/bloom-3b/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/bigscience/bloom-3b/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/bigscience/bloom-3b/resolve/main/special_tokens_map.json"
        ),
        "weight": hf_endpoint_url(
            "/bigscience/bloom-3b/resolve/main/pytorch_model.bin"
        ),
    },
    "bloom-7b1": {
        "config": hf_endpoint_url("/bigscience/bloom-7b1/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/bigscience/bloom-7b1/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/bigscience/bloom-7b1/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/bigscience/bloom-7b1/resolve/main/special_tokens_map.json"
        ),
        "weight": [
            hf_endpoint_url(
                "/bigscience/bloom-7b1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
            )
            for i in range(1, 3)
        ],
    },
    "bloomz-560m": {
        "config": hf_endpoint_url("/bigscience/bloomz-560m/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/bigscience/bloomz-560m/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/bigscience/bloomz-560m/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/bigscience/bloomz-560m/resolve/main/special_tokens_map.json"
        ),
        "weight": hf_endpoint_url(
            "/bigscience/bloomz-560m/resolve/main/pytorch_model.bin"
        ),
    },
    "bloomz-3b": {
        "config": hf_endpoint_url("/bigscience/bloomz-3b/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/bigscience/bloomz-3b/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/bigscience/bloomz-3b/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/bigscience/bloomz-3b/resolve/main/special_tokens_map.json"
        ),
        "weight": hf_endpoint_url(
            "/bigscience/bloomz-3b/resolve/main/pytorch_model.bin"
        ),
    },
    "bloomz-7b1": {
        "config": hf_endpoint_url("/bigscience/bloomz-7b1/resolve/main/config.json"),
        "tokenizer": hf_endpoint_url(
            "/bigscience/bloomz-7b1/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/bigscience/bloomz-7b1/resolve/main/tokenizer_config.json"
        ),
        "special_tokens_map": hf_endpoint_url(
            "/bigscience/bloomz-7b1/resolve/main/special_tokens_map.json"
        ),
        "weight": hf_endpoint_url(
            "/bigscience/bloomz-7b1/resolve/main/pytorch_model.bin"
        ),
    },
}

pretrained_bloom_extensions_infos = {}

import unitorch.cli.models.bloom.modeling
import unitorch.cli.models.bloom.processing
