# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_diffusers_available
import unitorch.modules.replace.datasets_v2

if is_diffusers_available():
    import unitorch.modules.replace.diffusers_v2
