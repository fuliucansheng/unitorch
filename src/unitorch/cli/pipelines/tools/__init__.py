# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_opencv_available

if is_opencv_available():
    from unitorch.cli.pipelines.tools.controlnet.canny import canny
    from unitorch.cli.pipelines.tools.controlnet.depth import depth
    from unitorch.cli.pipelines.tools.controlnet.dwpose import dwpose
    from unitorch.cli.pipelines.tools.controlnet.hed import hed
    from unitorch.cli.pipelines.tools.controlnet.lineart import lineart
    from unitorch.cli.pipelines.tools.controlnet.lineart_anime import lineart_anime

    controlnet_processes = {
        "canny": canny,
        "depth": depth,
        "dwpose": dwpose,
        "hed": hed,
        "lineart": lineart,
        "lineart_anime": lineart_anime,
        "inpainting": lambda x: x,
    }

    adapter_processes = {
        "canny": canny,
        "depth": depth,
        "dwpose": dwpose,
        "hed": hed,
        "lineart": lineart,
        "lineart_anime": lineart_anime,
        "inpainting": lambda x: x,
    }
else:
    controlnet_processes = {}
    adapter_processes = {}
