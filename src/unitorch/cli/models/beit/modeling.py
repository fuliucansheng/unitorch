# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.beit import (
    BeitForImageClassification as _BeitForImageClassification,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.beit import pretrained_beit_infos


@register_model("core/model/classification/beit")
class BeitForImageClassification(_BeitForImageClassification):
    """Class for image classification using the Beit model."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
    ):
        """
        Initialize BeitForImageClassification.

        Args:
            config_path (str): The path to the model configuration file.
            num_classes (int, optional): The number of output classes. Defaults to 1.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/beit")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BeitForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            BeitForImageClassification: An instance of BeitForImageClassification.
        """
        config.set_default_section("core/model/classification/beit")
        pretrained_name = config.getoption("pretrained_name", "default-beit")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_beit_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(
            config_path=config_path,
            num_classes=num_classes,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_beit_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the BeitForImageClassification model.

        Args:
            pixel_values (torch.Tensor): The input pixel values.

        Returns:
            ClassificationOutputs: The model outputs.
        """
        outputs = super().forward(pixel_values=pixel_values)
        return ClassificationOutputs(outputs=outputs)
