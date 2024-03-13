# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.optims.lion import Lion
from unitorch.models import CheckpointMixin
from unitorch.cli import add_default_section_for_init, register_optim


@register_optim("core/optim/lion")
class LionOptimizer(Lion, CheckpointMixin):
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
    @add_default_section_for_init("core/optim/lion")
    def from_core_configure(cls, config, **kwargs):
        pass
