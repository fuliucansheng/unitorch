# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import (
    is_diffusers_available,
    is_opencv_available,
    is_vllm_available,
)
import unitorch.cli.fastapis.info
import unitorch.cli.fastapis.bria
import unitorch.cli.fastapis.clip
import unitorch.cli.fastapis.detr
import unitorch.cli.fastapis.dpt
import unitorch.cli.fastapis.gemma
import unitorch.cli.fastapis.gemma_vl
import unitorch.cli.fastapis.grounding_dino
import unitorch.cli.fastapis.llama
import unitorch.cli.fastapis.llava
import unitorch.cli.fastapis.mask2former
import unitorch.cli.fastapis.mistral
import unitorch.cli.fastapis.qwen
import unitorch.cli.fastapis.qwen_vl
import unitorch.cli.fastapis.sam
import unitorch.cli.fastapis.segformer
import unitorch.cli.fastapis.siglip
import unitorch.cli.fastapis.servers

if is_vllm_available():
    import unitorch.cli.fastapis.qwen_vllm
    import unitorch.cli.fastapis.qwen_vl_vllm

if is_diffusers_available():
    import unitorch.cli.fastapis.qwen_image
    import unitorch.cli.fastapis.flux2

    if is_opencv_available():
        import unitorch.cli.fastapis.wan
        import unitorch.cli.fastapis.lucy
