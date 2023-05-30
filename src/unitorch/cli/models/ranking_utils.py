# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models.modeling_utils import TensorsOutputs, TensorsTargets


@dataclass
class RankingOutputs(TensorsOutputs, WriterMixin):
    outputs: torch.Tensor


@dataclass
class RankingTargets(TensorsTargets):
    targets: torch.Tensor
    masks: Optional[torch.Tensor] = torch.empty(0)
