# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import math
import random
import logging
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.models.blip_2.modeling_blip_2 import (
    Blip2Config,
    Blip2Model,
    Blip2ForConditionalGeneration,
)
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.clip.modeling import AllGather
