# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericOnnxModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models import TensorsOutputs, TensorsTargets, ACT2FN


@dataclass
class PandasOutputs(WriterMixin):
    outputs: pd.DataFrame


def onnx_model_decorator(cls):
    class OnnxModel(GenericOnnxModel):
        def __init__(self, *args, **kwargs):
            super().__init__()

    return OnnxModel
