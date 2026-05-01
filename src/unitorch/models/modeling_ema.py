# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import logging
import math
import os
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


class ExponentialMovingAverage(nn.Module):
    """Exponential Moving Average (EMA) wrapper for a model's parameters.

    The effective decay at step *t* is ``decay * (1 - exp(-t / tau))``,
    which ramps from 0 up to *decay* so early updates are not over-smoothed.
    """

    checkpoint_name = "pytorch_ema_model.bin"

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: int = 2000,
        num_steps: int = 0,
    ) -> None:
        super().__init__()
        self.model = deepcopy(model)
        self.num_steps = num_steps
        self._decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

        for p in self.model.parameters():
            p.requires_grad_(False)

    def from_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Load EMA weights from *ckpt_dir*."""
        weight_name = weight_name or self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
        logging.info("%s loaded weights from %s", type(self).__name__, weight_path)

    def save_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Save EMA weights to *ckpt_dir*."""
        weight_name = weight_name or self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(self.model.state_dict(), weight_path)
        logging.info("%s saved checkpoint to %s", type(self).__name__, weight_path)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the EMA model."""
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        """Update EMA parameters with one step from *model*."""
        self.num_steps += 1
        rate = self._decay_fn(self.num_steps)
        new_state = model.state_dict()
        for key, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(rate).add_(new_state[key].detach(), alpha=1 - rate)
