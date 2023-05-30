# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random

from unitorch.utils import pop_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import ACT2FN
from unitorch.cli.models import (
    ModelInputs,
    ModelOutputs,
    ModelTargets,
    TensorsInputs,
    ListTensorsInputs,
    EmbeddingOutputs,
)


def _process_returns(kwargs, dtype="TensorsInputs"):
    assert dtype in globals()
    cls = globals()[dtype]
    assert (
        issubclass(cls, ModelInputs)
        or issubclass(cls, ModelOutputs)
        or issubclass(cls, ModelTargets)
    )
    return cls(**kwargs)


class PreProcessor:
    def __init__(
        self,
    ):
        pass

    @classmethod
    @add_default_section_for_init("core/process")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("core/process/number")
    def _number(
        self,
        text: Union[int, float, str],
        dtype: Optional[str] = "int",
        key: Optional[str] = "num",
        returns: Optional[str] = "TensorsInputs",
    ):
        num = torch.tensor(float(text))
        if dtype == "int":
            num = num.int()
        return _process_returns({key: num}, dtype=returns)

    @register_process("core/process/features")
    def _features(
        self,
        features: Union[List, str],
        sep: Optional[str] = None,
        pad: Optional[float] = None,
        num: Optional[int] = None,
        dtype: Optional[str] = "int",
        shape: Optional[tuple] = None,
        key: Optional[str] = "features",
        returns: Optional[str] = "TensorsInputs",
    ):
        if isinstance(features, str):
            features = features.split(sep=sep)

        features = torch.tensor(list(map(float, features)))
        if dtype == "int":
            features = features.int()

        if pad is not None and num is not None:
            features = features[:num]
            padding = torch.zeros(num - len(features)).to(features) + pad
            features = torch.cat([features, padding])

        if shape is not None:
            features = features.reshape(shape)
        return _process_returns({key: features}, dtype=returns)


class PostProcessor:
    def __init__(
        self,
        act_fn: Optional[str] = None,
    ):
        self.act_fn = ACT2FN.get(act_fn, None)

    @classmethod
    @add_default_section_for_init("core/postprocess")
    def from_core_configure(cls, config, **kwargs):
        pass
