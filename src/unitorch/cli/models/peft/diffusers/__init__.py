# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch.cli.models.peft.diffusers.modeling_qwen_image
import unitorch.cli.models.peft.diffusers.modeling_wan
import unitorch.cli.models.peft.diffusers.modeling_flux2
import unitorch.cli.models.peft.diffusers.modeling_lucy

from unitorch.utils.import_utils import is_flux3_available

if is_flux3_available():
    import unitorch.cli.models.peft.diffusers.modeling_flux3
