# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import math
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler.LambdaLR):
    """LR scheduler with a linear warm-up phase followed by a cosine decay.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Number of steps for the linear warm-up.
        num_training_steps: Total number of training steps.
        num_cycles: Number of cosine half-cycles. Defaults to ``0.5`` (one
                    half-cycle, i.e. decay to zero).
        last_epoch: Index of the last epoch. Defaults to ``-1``.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            progress = float(step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

        super().__init__(optimizer, lr_lambda, last_epoch)


class LinearWarmupScheduler(optim.lr_scheduler.LambdaLR):
    """LR scheduler with a linear warm-up phase followed by a linear decay.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Number of steps for the linear warm-up.
        num_training_steps: Total number of training steps.
        last_epoch: Index of the last epoch. Defaults to ``-1``.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        super().__init__(optimizer, lr_lambda, last_epoch)
