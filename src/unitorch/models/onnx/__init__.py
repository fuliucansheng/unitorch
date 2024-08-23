# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import abc
import warnings
import logging
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import replace, is_onnxruntime_available
from unitorch.models import GenericModel


class GenericOnnxModel(GenericModel):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def postprocess(self, *args, **kwargs):
        pass


class SimpleOnnxModel(GenericOnnxModel):
    def __init__(self):
        super().__init__()

    def setup(self):
        pass

    def preprocess(self):
        pass

    def forward(self):
        pass

    def postprocess(self):
        pass
