# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.siglip2 import (
    Siglip2ForPretrain as _Siglip2ForPretrain,
    Siglip2ForClassification as _Siglip2ForClassification,
    Siglip2ForTextClassification as _Siglip2ForTextClassification,
    Siglip2ForImageClassification as _Siglip2ForImageClassification,
    Siglip2ForMatching as _Siglip2ForMatching,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.siglip2 import pretrained_siglip2_infos


@register_model("core/model/pretrain/siglip2")
class Siglip2ForPretrain(_Siglip2ForPretrain):
    """CLIP model for pretraining."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        """
        Initialize the Siglip2ForPretrain model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_all_gather (bool, optional): Whether to use all_gather operation. Defaults to True.
        """
        super().__init__(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )

    @classmethod
    @add_default_section_for_init("core/model/pretrain/siglip2")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Siglip2ForPretrain from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Siglip2ForPretrain: An instance of the Siglip2ForPretrain model.
        """
        config.set_default_section("core/model/pretrain/siglip2")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip2-base-patch16-224"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "config"),
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
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "weight"),
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
        pixel_masks: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Input pixel values.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            LossOutputs: The loss outputs.
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
            spatial_shapes=spatial_shapes,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return LossOutputs(loss=outputs)


@register_model("core/model/classification/siglip2")
class Siglip2ForClassification(_Siglip2ForClassification):
    """CLIP model for classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the Siglip2ForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            num_classes (int, optional): The number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/siglip2")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Siglip2ForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Siglip2ForClassification: An instance of the Siglip2ForClassification model.
        """
        config.set_default_section("core/model/classification/siglip2")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip2-base-patch16-224"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "config"),
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
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "weight"),
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
        pixel_masks: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Input pixel values.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
            spatial_shapes=spatial_shapes,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/siglip2/text")
class Siglip2ForTextClassification(_Siglip2ForTextClassification):
    """CLIP model for text classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the Siglip2ForTextClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            num_classes (int, optional): The number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/siglip2/text")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Siglip2ForTextClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Siglip2ForTextClassification: An instance of the Siglip2ForTextClassification model.
        """
        config.set_default_section("core/model/classification/siglip2/text")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip2-base-patch16-224"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "config"),
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
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "weight"),
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
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/classification/siglip2/image")
class Siglip2ForImageClassification(_Siglip2ForImageClassification):
    """CLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the Siglip2ForImageClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            num_classes (int, optional): The number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/siglip2/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Siglip2ForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Siglip2ForImageClassification: An instance of the Siglip2ForImageClassification model.
        """
        config.set_default_section("core/model/classification/siglip2/image")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip2-base-patch16-224"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "config"),
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
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ):
        """
        Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        outputs = super().forward(
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
            spatial_shapes=spatial_shapes,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/matching/siglip2")
class Siglip2ForMatching(_Siglip2ForMatching):
    """CLIP model for classification."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the Siglip2ForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            num_classes (int, optional): The number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/matching/siglip2")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Siglip2ForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Siglip2ForClassification: An instance of the Siglip2ForClassification model.
        """
        config.set_default_section("core/model/matching/siglip2")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip2-base-patch16-224"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "config"),
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
            nested_dict_value(pretrained_siglip2_infos, pretrained_name, "weight"),
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
        pixel_masks: torch.Tensor,
        spatial_shapes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Input pixel values.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
            spatial_shapes=spatial_shapes,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)
