# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.siglip import (
    SiglipForPretrain as _SiglipForPretrain,
    SiglipForClassification as _SiglipForClassification,
    SiglipForTextClassification as _SiglipForTextClassification,
    SiglipForImageClassification as _SiglipForImageClassification,
    SiglipForMatching as _SiglipForMatching,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.siglip import pretrained_siglip_infos


@register_model("core/model/pretrain/siglip")
class SiglipForPretrain(_SiglipForPretrain):
    """Siglip model for pretraining."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )

    @classmethod
    @add_default_section_for_init("core/model/pretrain/siglip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/pretrain/siglip")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
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
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return LossOutputs(loss=outputs)


@register_model("core/model/classification/siglip")
class SiglipForClassification(_SiglipForClassification):
    """Siglip model for multimodal classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/siglip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/siglip")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
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
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/siglip/text")
class SiglipForTextClassification(_SiglipForTextClassification):
    """Siglip model for text classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/siglip/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/siglip/text")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_Truemodel", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
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
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/siglip/image")
class SiglipForImageClassification(_SiglipForImageClassification):
    """Siglip model for image-only classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/siglip/image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/classification/siglip/image")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
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


@register_model("core/model/matching/siglip")
class SiglipForMatching(_SiglipForMatching):
    """Siglip model for image-text matching."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/matching/siglip")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/matching/siglip")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
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
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)
