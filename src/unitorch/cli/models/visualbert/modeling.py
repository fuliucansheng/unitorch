# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.visualbert import (
    VisualBertForClassification as _VisualBertForClassification,
)
from unitorch.models.visualbert import VisualBertForPretrain as _VisualBertForPretrain
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.visualbert import pretrained_visualbert_infos


@register_model("core/model/classification/visualbert")
class VisualBertForClassification(_VisualBertForClassification):
    """VisualBERT for Classification model."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize VisualBertForClassification.

        Args:
            config_path (str): The path to the model's configuration file.
            num_classes (Optional[int]): The number of classes for classification.
                Defaults to 1.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing
                to save memory during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/visualbert")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of VisualBertForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            VisualBertForClassification: The initialized VisualBertForClassification instance.
        """
        config.set_default_section("core/model/classification/visualbert")
        pretrained_name = config.getoption("pretrained_name", "default-visualbert")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_visualbert_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(config_path, num_classes, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_visualbert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        visual_token_type_ids: torch.Tensor,
    ):
        """
        Forward pass of the VisualBertForClassification model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.
            position_ids (torch.Tensor): The position IDs.
            visual_embeds (torch.Tensor): The visual embeddings.
            visual_attention_mask (torch.Tensor): The visual attention mask.
            visual_token_type_ids (torch.Tensor): The visual token type IDs.

        Returns:
            ClassificationOutputs: The output logits of the model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/pretrain/visualbert")
class VisualBertForPretrain(_VisualBertForPretrain):
    """VisualBERT for Pretraining model."""

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize VisualBertForPretrain.

        Args:
            config_path (str): The path to the model's configuration file.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing
                to save memory during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/pretrain/visualbert")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of VisualBertForPretrain from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            VisualBertForPretrain: The initialized VisualBertForPretrain instance.
        """
        config.set_default_section("core/model/pretrain/visualbert")
        pretrained_name = config.getoption("pretrained_name", "default-visualbert")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_visualbert_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_visualbert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        visual_token_type_ids: torch.Tensor,
        nsp_label: torch.Tensor,
        mlm_label: torch.Tensor,
        mlm_label_mask: torch.Tensor,
    ):
        """
        Forward pass of the VisualBertForPretrain model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.
            position_ids (torch.Tensor): The position IDs.
            visual_embeds (torch.Tensor): The visual embeddings.
            visual_attention_mask (torch.Tensor): The visual attention mask.
            visual_token_type_ids (torch.Tensor): The visual token type IDs.
            nsp_label (torch.Tensor): The next sentence prediction label.
            mlm_label (torch.Tensor): The masked language model label.
            mlm_label_mask (torch.Tensor): The masked language model label mask.

        Returns:
            LossOutputs: The output loss of the model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            nsp_label=nsp_label,
            mlm_label=mlm_label,
            mlm_label_mask=mlm_label_mask,
        )
        return LossOutputs(loss=outputs)
