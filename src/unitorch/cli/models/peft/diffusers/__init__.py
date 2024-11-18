# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch.cli.models.peft.diffusers.modeling_controlnet
import unitorch.cli.models.peft.diffusers.modeling_controlnet_xl
import unitorch.cli.models.peft.diffusers.modeling_controlnet_3
import unitorch.cli.models.peft.diffusers.modeling_controlnet_flux
import unitorch.cli.models.peft.diffusers.modeling_stable
import unitorch.cli.models.peft.diffusers.modeling_stable_xl
import unitorch.cli.models.peft.diffusers.modeling_stable_3
import unitorch.cli.models.peft.diffusers.modeling_stable_flux
import unitorch.cli.models.peft.diffusers.modeling_adapter
import unitorch.cli.models.peft.diffusers.modeling_adapter_xl
from unitorch.cli.models.peft.diffusers.modeling_controlnet import (
    ControlNetLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_controlnet_xl import (
    ControlNetXLLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_controlnet_3 import (
    ControlNet3LoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_controlnet_flux import (
    ControlNetFluxLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_stable import (
    StableLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_stable_xl import (
    StableXLLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_stable_3 import (
    Stable3LoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_stable_flux import (
    StableFluxLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_adapter import (
    StableAdapterLoraForText2ImageGeneration,
)
from unitorch.cli.models.peft.diffusers.modeling_adapter_xl import (
    StableXLAdapterLoraForText2ImageGeneration,
)
