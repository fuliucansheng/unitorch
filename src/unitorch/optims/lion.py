# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum) optimizer.

    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Initialise Lion.

        Args:
            params: Iterable of parameters or parameter-group dicts.
            lr: Learning rate. Must be ≥ 0.
            betas: Coefficients ``(β₁, β₂)`` for the update interpolation and
                   the momentum EMA. Both must be in ``[0, 1)``.
            weight_decay: Decoupled weight-decay coefficient.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")

        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform a single optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                     the loss.

        Returns:
            Loss tensor if a closure was provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data.mul_(1 - lr * wd)

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
