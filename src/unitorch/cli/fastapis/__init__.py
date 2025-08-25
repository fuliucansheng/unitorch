# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_diffusers_available
import unitorch.cli.fastapis.info
import unitorch.cli.fastapis.bria
import unitorch.cli.fastapis.qwen
import unitorch.cli.fastapis.qwen_vl
import unitorch.cli.fastapis.llava

if is_diffusers_available():
    import unitorch.cli.fastapis.stable
    import unitorch.cli.fastapis.controlnet
    import unitorch.cli.fastapis.stable_xl
    import unitorch.cli.fastapis.controlnet_xl
    import unitorch.cli.fastapis.stable_3
    import unitorch.cli.fastapis.controlnet_3
    import unitorch.cli.fastapis.stable_flux
    import unitorch.cli.fastapis.controlnet_flux
    import unitorch.cli.fastapis.qwen_image
    import unitorch.cli.fastapis.wan
