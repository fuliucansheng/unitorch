# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Dict, List, Optional, Union

from unitorch.utils import pop_value
from unitorch.cli import (
    config_defaults_init,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import ACT2FN
from unitorch.cli.models import (
    ModelInputs,
    ModelOutputs,
    ModelTargets,
    TensorInputs,
    TensorSeqInputs,
    EmbeddingOutputs,
)


def _process_returns(kwargs, dtype="TensorInputs"):
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
        map_dict: Optional[Dict[str, str]] = None,
    ):
        self.map_dict = map_dict if map_dict is not None else {}

    @classmethod
    @config_defaults_init("core/process")
    def from_config(cls, config, **kwargs):
        pass

    @register_process("core/process/number")
    def _number(
        self,
        text: Union[int, float, str],
        dtype: Optional[str] = "int",
        key: Optional[str] = "num",
        returns: Optional[str] = "TensorInputs",
    ):
        num = torch.tensor(float(text))
        if dtype == "int":
            num = num.int()
        return _process_returns({key: num}, dtype=returns)

    @register_process("core/process/map")
    def _map(
        self,
        text: str,
        sep: Optional[str] = None,
        default: Optional[str] = None,
    ):
        if sep is not None:
            text = text.split(sep=sep)
        if isinstance(text, list):
            res = [self.map_dict.get(t, default) for t in text]
        else:
            res = self.map_dict.get(text, default)
        return res

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
        returns: Optional[str] = "TensorInputs",
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
    @config_defaults_init("core/postprocess")
    def from_config(cls, config, **kwargs):
        pass
