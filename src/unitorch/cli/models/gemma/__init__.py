# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from functools import lru_cache
from typing import Optional

from huggingface_hub import hf_hub_download

from unitorch import get_cache_dir

pretrained_gemma_infos = {
    "gemma-4-12b": {
        "repo_id": "google/gemma-4-12B",
        "revision": "main",
        "config": "config.json",
        "tokenizer": "tokenizer.json",
        "tokenizer_config": "tokenizer_config.json",
        "processor_config": "processor_config.json",
        "weight": "model.safetensors",
    },
}

pretrained_gemma_extensions_infos = {}


@lru_cache(maxsize=None)
def resolve_pretrained_gemma_path(
    pretrained_name: str,
    key: str,
) -> Optional[str]:
    info = pretrained_gemma_infos[pretrained_name]
    filename = info.get(key)
    if filename is None:
        return None

    return hf_hub_download(
        repo_id=info["repo_id"],
        filename=filename,
        revision=info.get("revision", "main"),
        cache_dir=get_cache_dir(),
    )


import unitorch.cli.models.gemma.modeling
import unitorch.cli.models.gemma.modeling_vl
import unitorch.cli.models.gemma.processing
import unitorch.cli.models.gemma.processing_vl
