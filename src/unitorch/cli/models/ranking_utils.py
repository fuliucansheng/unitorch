# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from dataclasses import dataclass
from typing import Optional
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models.modeling_utils import TensorOutputs, TensorTargets


@dataclass
class RankingOutputs(TensorOutputs, WriterMixin):
    outputs: torch.Tensor


@dataclass
class RankingTargets(TensorTargets):
    targets: torch.Tensor
    masks: Optional[torch.Tensor] = torch.empty(0)
