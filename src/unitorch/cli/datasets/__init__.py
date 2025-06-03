# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from unitorch.utils import is_megatron_available

import unitorch.cli.datasets.hf

if is_megatron_available():
    import unitorch.cli.datasets.megatron
