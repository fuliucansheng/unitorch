# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from unitorch.utils import pop_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models.classification_utils import ClassificationTargets


class LabelProcessor:
    """
    Processor for label-related operations.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        sep: Optional[str] = ",",
        max_seq_length: Optional[int] = 128,
        map_dict: Optional[Dict] = dict(),
    ):
        """
        Initializes a new instance of the LabelProcessor.

        Args:
            num_classes (Optional[int]): The number of classes. Defaults to None.
            sep (Optional[str]): The separator used for splitting text. Defaults to ",".
            max_seq_length (Optional[int]): The maximum sequence length. Defaults to 128.
            map_dict (Optional[Dict]): A dictionary for mapping labels. Defaults to an empty dictionary.
        """
        self.num_classes = num_classes
        self.sep = sep
        self.map_dict = map_dict

    @classmethod
    @add_default_section_for_init("core/process/label")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a new instance of the LabelProcessor using the configuration from the core.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the LabelProcessor.
        """
        pass

    @register_process("core/process/label")
    def _label(
        self,
        text: Union[int, float, str],
        dtype: Optional[str] = "int",
    ):
        """
        Processes a single label.

        Args:
            text (Union[int, float, str]): The label to process.
            dtype (Optional[str]): The desired data type of the output. Defaults to "int".

        Returns:
            A ClassificationTargets object containing the processed label.
        """
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
        """
        Processes a sequence of labels.

        Args:
            text (Union[List, str]): The sequence of labels to process.
            sep (Optional[str]): The separator used for splitting text. Defaults to None.
            dtype (Optional[str]): The desired data type of the output. Defaults to "int".

        Returns:
            A ClassificationTargets object containing the processed sequence of labels.
        """
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
        """
        Processes binary labels.

        Args:
            text (Union[List, str]): The binary labels to process.
            sep (Optional[str]): The separator used for splitting text. Defaults to None.

        Returns:
            A ClassificationTargets object containing the processed binary labels.
        """
        outputs = torch.zeros(self.num_classes)
        if isinstance(text, str):
            sep = pop_value(sep, self.sep)
            indexes = [int(self.map_dict.get(t, t)) for t in text.split(sep)]
        else:
            indexes = [int(t) for t in text]
        outputs[indexes] = 1
        return ClassificationTargets(targets=outputs)
