# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from diffusers.schedulers import (
    DPMSolverSDEScheduler,
    UniPCMultistepScheduler,
)

Schedulers = {
    "DPM++SDE": DPMSolverSDEScheduler,
    "UniPC++": UniPCMultistepScheduler,
}

import unitorch.cli.pipelines.animate
import unitorch.cli.pipelines.bloom
import unitorch.cli.pipelines.detr
import unitorch.cli.pipelines.peft
import unitorch.cli.pipelines.stable
import unitorch.cli.pipelines.stable_xl
import unitorch.cli.pipelines.stable_xl_refiner
import unitorch.cli.pipelines.controlnet
import unitorch.cli.pipelines.controlnet_xl
import unitorch.cli.pipelines.llama
import unitorch.cli.pipelines.mistral
import unitorch.cli.pipelines.sam
