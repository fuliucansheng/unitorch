# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from diffusers.schedulers import (
    DPMSolverSDEScheduler,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)

Schedulers = {
    "DPM++SDE": DPMSolverSDEScheduler,
    "UniPC++": UniPCMultistepScheduler,
    "FlowMatchEuler": FlowMatchEulerDiscreteScheduler,
    "EulerAncestral": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
}

import unitorch.cli.pipelines.blip
import unitorch.cli.pipelines.bloom
import unitorch.cli.pipelines.bria
import unitorch.cli.pipelines.detr
import unitorch.cli.pipelines.dpt
import unitorch.cli.pipelines.stable
import unitorch.cli.pipelines.stable_xl
import unitorch.cli.pipelines.stable_3
import unitorch.cli.pipelines.llama
import unitorch.cli.pipelines.mistral
import unitorch.cli.pipelines.sam
