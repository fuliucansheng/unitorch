# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging
import json
import numpy as np
from unitorch import (
    is_deepspeed_available,
    is_megatron_available,
)


import unitorch.cli.tasks.supervised

if is_deepspeed_available():
    import unitorch.cli.tasks.deepspeed

if is_megatron_available():
    import unitorch.cli.tasks.megatron
