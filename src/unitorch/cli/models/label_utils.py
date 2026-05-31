# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Dict, List, Optional, Union

from unitorch.utils import pop_value
from unitorch.cli import (
    config_defaults_init,
    register_process,
)
from unitorch.cli.models.classification_utils import ClassificationTargets


class LabelProcessor:
    """Processor for label-related operations."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        sep: Optional[str] = ",",
        max_seq_length: Optional[int] = 128,
        map_dict: Optional[Dict] = None,
    ):
        self.num_classes = num_classes
        self.sep = sep
        self.max_seq_length = max_seq_length
        self.map_dict = map_dict if map_dict is not None else {}

    @classmethod
    @config_defaults_init("core/process/label")
    def from_config(cls, config, **kwargs):
        pass

    @register_process("core/process/label")
    def _label(
        self,
        text: Union[int, float, str],
        dtype: Optional[str] = "int",
    ):
        """Convert a scalar label to a ClassificationTargets tensor."""
        if text in self.map_dict:
            text = self.map_dict[text]

        if dtype == "int":
            outputs = torch.tensor(int(text))
        else:
            outputs = torch.tensor(float(text))
        return ClassificationTargets(targets=outputs)

    @register_process("core/process/label/sequence")
    def _sequence_label(
        self,
        text: Union[List, str],
        sep: Optional[str] = None,
        dtype: Optional[str] = "int",
    ):
        """Convert a sequence of labels to a ClassificationTargets tensor."""
        if isinstance(text, str):
            sep = pop_value(sep, self.sep)
            tensor = [float(self.map_dict.get(t, t)) for t in text.split(sep)]
        else:
            tensor = [float(t) for t in text]

        if dtype == "int":
            outputs = torch.tensor(tensor).int()
        else:
            outputs = torch.tensor(tensor)
        return ClassificationTargets(targets=outputs)

    @register_process("core/process/label/binary")
    def _binary_label(
        self,
        text: Union[List, str],
        sep: Optional[str] = None,
    ):
        """Build a multi-hot binary ClassificationTargets tensor of length num_classes."""
        outputs = torch.zeros(self.num_classes)
        if isinstance(text, str):
            sep = pop_value(sep, self.sep)
            indexes = [int(self.map_dict.get(t, t)) for t in text.split(sep)]
        else:
            indexes = [int(t) for t in text]
        outputs[indexes] = 1
        return ClassificationTargets(targets=outputs)
