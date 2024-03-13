# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import AdamW
from unitorch.optims.lion import Lion

from unitorch.utils import is_bitsandbytes_available

if is_bitsandbytes_available():
    from bitsandbytes.optim import Adam8bit, AdamW8bit
