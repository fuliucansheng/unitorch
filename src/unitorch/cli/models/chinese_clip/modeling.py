# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Dict, List, Optional, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.chinese_clip import (
    ChineseClipForPretrain as _ChineseClipForPretrain,
    ChineseClipForClassification as _ChineseClipForClassification,
    ChineseClipForTextClassification as _ChineseClipForTextClassification,
    ChineseClipForImageClassification as _ChineseClipForImageClassification,
)
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.chinese_clip import pretrained_chinese_clip_infos


@register_model("core/model/pretrain/chinese_clip")
class ChineseClipForPretrain(_ChineseClipForPretrain):
    """Chinese CLIP model for pretraining."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )

    @classmethod
    @config_defaults_init("core/model/pretrain/chinese_clip")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/pretrain/chinese_clip")
        pretrained_name = config.getoption(
            "pretrained_name", "chinese-clip-vit-base-patch16"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return LossOutputs(loss=outputs)


@register_model("core/model/classification/chinese_clip")
class ChineseClipForClassification(_ChineseClipForClassification):
    """Chinese CLIP model for classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @config_defaults_init("core/model/classification/chinese_clip")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/classification/chinese_clip")
        pretrained_name = config.getoption(
            "pretrained_name", "chinese-clip-vit-base-patch16"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/chinese_clip/text")
class ChineseClipForTextClassification(_ChineseClipForTextClassification):
    """Chinese CLIP model for text classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @config_defaults_init("core/model/classification/chinese_clip/text")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/classification/chinese_clip/text")
        pretrained_name = config.getoption(
            "pretrained_name", "chinese-clip-vit-base-patch16"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_Truemodel", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids=None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/chinese_clip/image")
class ChineseClipForImageClassification(_ChineseClipForImageClassification):
    """Chinese CLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @config_defaults_init("core/model/classification/chinese_clip/image")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/classification/chinese_clip/image")
        pretrained_name = config.getoption(
            "pretrained_name", "chinese-clip-vit-base-patch16"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_chinese_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        outputs = super().forward(pixel_values=pixel_values)
        return ClassificationOutputs(outputs=outputs)
