# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import math
import torch.optim as optim
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


class CosineWarmupScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: Optional[float] = 0.5,
        last_epoch: Optional[int] = -1,
    ):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )

        super().__init__(optimizer, lr_lambda, last_epoch)


class LinearWarmupScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: Optional[int] = -1,
    ):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        super().__init__(optimizer, lr_lambda, last_epoch)
