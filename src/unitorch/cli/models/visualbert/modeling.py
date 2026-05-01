# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from torch import autocast
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
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/visualbert")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/visualbert")
        pretrained_name = config.getoption("pretrained_name", "visualbert-vqa-coco-pre")
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

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
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
        super().__init__(
            config_path=config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/pretrain/visualbert")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/pretrain/visualbert")
        pretrained_name = config.getoption("pretrained_name", "visualbert-vqa-coco-pre")
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

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
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
