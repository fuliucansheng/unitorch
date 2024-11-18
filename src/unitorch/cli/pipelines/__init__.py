# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_diffusers_available

if is_diffusers_available():
    from diffusers.schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        PNDMScheduler,
        DEISMultistepScheduler,
        DPMSolverSDEScheduler,
        UniPCMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        FlowMatchEulerDiscreteScheduler,
    )

    Schedulers = {
        "DPM++SDE": DPMSolverSDEScheduler,
        "UniPC++": UniPCMultistepScheduler,
        "DEIS++": DEISMultistepScheduler,
        "DDIM": DDIMScheduler,
        "DDPM": DDPMScheduler,
        "PNDM": PNDMScheduler,
        "FlowMatchEuler": FlowMatchEulerDiscreteScheduler,
        "EulerAncestral": EulerAncestralDiscreteScheduler,
        "Euler": EulerDiscreteScheduler,
    }
    import unitorch.cli.pipelines.stable
    import unitorch.cli.pipelines.stable_xl

    import unitorch.cli.pipelines.stable_3
    import unitorch.cli.pipelines.stable_flux
else:
    Schedulers = {}

import unitorch.cli.pipelines.blip
import unitorch.cli.pipelines.bloom
import unitorch.cli.pipelines.bria
import unitorch.cli.pipelines.detr
import unitorch.cli.pipelines.dpt
import unitorch.cli.pipelines.grounding_dino
import unitorch.cli.pipelines.segformer
import unitorch.cli.pipelines.llama
import unitorch.cli.pipelines.llava
import unitorch.cli.pipelines.mask2former
import unitorch.cli.pipelines.mistral
import unitorch.cli.pipelines.sam
