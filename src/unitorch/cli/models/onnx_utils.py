# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import pandas as pd
from dataclasses import dataclass
from unitorch.models import GenericOnnxModel
from unitorch.cli import WriterMixin


@dataclass
class PandasOutputs(WriterMixin):
    outputs: pd.DataFrame


def onnx_model_decorator(cls):
    class OnnxModel(GenericOnnxModel):
        def __init__(self, *args, **kwargs):
            super().__init__()

    return OnnxModel
