# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.deberta import (
    DebertaForClassification as _DebertaForClassification,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.deberta import pretrained_deberta_infos


@register_model("core/model/classification/deberta")
class DebertaForClassification(_DebertaForClassification):
    """Deberta model for classification tasks."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the DebertaForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            num_classes (int, optional): The number of classes for classification. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/deberta")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of DebertaForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            DebertaForClassification: An instance of DebertaForClassification.
        """
        config.set_default_section("core/model/classification/deberta")
        pretrained_name = config.getoption("pretrained_name", "default-deberta")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_deberta_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(config_path, num_classes, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_deberta_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass on the DebertaForClassification model.

        Args:
            input_ids (torch.Tensor): The input tensor containing the input IDs.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            token_type_ids (torch.Tensor, optional): The token type IDs tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor. Defaults to None.

        Returns:
            ClassificationOutputs: The output of the classification model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)
