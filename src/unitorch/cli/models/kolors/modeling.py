# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.kolors import (
    KolorsMPSModel as _KolorsMPSModel,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.kolors import pretrained_kolors_infos


@register_model("core/model/classification/kolors/mps")
class KolorsMPSModel(_KolorsMPSModel):
    """CLIP model for classification."""

    def __init__(self, config_path: str):
        """
        Initialize the ClipForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            num_classes (int, optional): The number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/kolors/mps")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForClassification: An instance of the ClipForClassification model.
        """
        config.set_default_section("core/model/classification/kolors/mps")
        pretrained_name = config.getoption("pretrained_name", "kolors-mps-overall")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_kolors_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        inst = cls(
            config_path=config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_kolors_infos, pretrained_name, "weight"),
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
        condition_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        condition_attention_mask: Optional[torch.Tensor] = None,
        condition_position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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
        if self.training:
            assert labels is not None
            outputs = super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                condition_input_ids=condition_input_ids,
                condition_attention_mask=condition_attention_mask,
                condition_position_ids=condition_position_ids,
                labels=labels,
            )
            return LossOutputs(outputs=outputs)

        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            condition_input_ids=condition_input_ids,
            condition_attention_mask=condition_attention_mask,
            condition_position_ids=condition_position_ids,
        )
        return ClassificationOutputs(outputs=outputs)
