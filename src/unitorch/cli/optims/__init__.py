# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.optims import SGD, Adam, AdamW
from unitorch.models import CheckpointMixin
from unitorch.cli import add_default_section_for_init, register_optim


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
    @add_default_section_for_init("core/optim/sgd")
    def from_core_configure(cls, config, **kwargs):
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
    @add_default_section_for_init("core/optim/adam")
    def from_core_configure(cls, config, **kwargs):
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
    @add_default_section_for_init("core/optim/adamw")
    def from_core_configure(cls, config, **kwargs):
        pass


# more optims
import unitorch.cli.optims.lion

# bitsandbytes
from unitorch.utils import is_bitsandbytes_available

if is_bitsandbytes_available():
    from unitorch.optims import Adam8bit, AdamW8bit

    @register_optim("core/optim/adam8bit")
    class Adam8bitOptimizer(Adam8bit, CheckpointMixin):
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
        @add_default_section_for_init("core/optim/adam8bit")
        def from_core_configure(cls, config, **kwargs):
            pass

    @register_optim("core/optim/adamw8bit")
    class AdamW8bitOptimizer(AdamW8bit, CheckpointMixin):
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
        @add_default_section_for_init("core/optim/adamw8bit")
        def from_core_configure(cls, config, **kwargs):
            pass
