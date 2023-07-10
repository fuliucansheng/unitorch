# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import sys
import logging
import json
import numpy as np
from unitorch import (
    is_deepspeed_available,
    is_accelerate_available,
    is_megatron_available,
)


import unitorch.cli.tasks.supervised
from unitorch.cli.tasks.supervised import SupervisedTask

if is_deepspeed_available():
    import unitorch.cli.tasks.deepspeed
    from unitorch.cli.tasks.deepspeed import DeepspeedTask
