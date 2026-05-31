# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional
from unitorch.optims import SGD, Adam, AdamW, Adafactor
from unitorch.models import CheckpointMixin
from unitorch.cli import config_defaults_init, register_optim


@register_optim("core/optim/sgd")
class SGDOptimizer(SGD, CheckpointMixin):
    def __init__(
        self,
        params,
        learning_rate: Optional[float] = 0.00001,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
        )

    @classmethod
    @config_defaults_init("core/optim/sgd")
    def from_config(cls, config, **kwargs):
        pass


@register_optim("core/optim/adam")
class AdamOptimizer(Adam, CheckpointMixin):
    def __init__(
        self,
        params,
        learning_rate: Optional[float] = 0.00001,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
        )

    @classmethod
    @config_defaults_init("core/optim/adam")
    def from_config(cls, config, **kwargs):
        pass


@register_optim("core/optim/adamw")
class AdamWOptimizer(AdamW, CheckpointMixin):
    def __init__(
        self,
        params,
        learning_rate: Optional[float] = 0.00001,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
        )

    @classmethod
    @config_defaults_init("core/optim/adamw")
    def from_config(cls, config, **kwargs):
        pass


@register_optim("core/optim/adafactor")
class AdafactorOptimizer(Adafactor, CheckpointMixin):
    def __init__(
        self,
        params,
        learning_rate: Optional[float] = 0.00001,
        scale_parameter: bool = False,
        relative_step: bool = False,
        warmup_init: bool = False,
    ):
        super().__init__(
            params=params,
            lr=learning_rate,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )

    @classmethod
    @config_defaults_init("core/optim/adafactor")
    def from_config(cls, config, **kwargs):
        pass


import unitorch.cli.optims.lion
