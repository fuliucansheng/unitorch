# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_diffusers_available, is_opencv_available
import unitorch.cli.fastapis.info
import unitorch.cli.fastapis.blip
import unitorch.cli.fastapis.bria
import unitorch.cli.fastapis.clip
import unitorch.cli.fastapis.detr
import unitorch.cli.fastapis.dpt
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

if is_diffusers_available():
    import unitorch.cli.fastapis.stable_flux
    import unitorch.cli.fastapis.qwen_image

    if is_opencv_available():
        import unitorch.cli.fastapis.wan
