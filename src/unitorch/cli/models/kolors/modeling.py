# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.kolors import (
    KolorsMPSModel as _KolorsMPSModel,
)
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.kolors import pretrained_kolors_infos


@register_model("core/model/classification/kolors/mps")
class KolorsMPSModel(_KolorsMPSModel):
    """Kolors MPS model for image-text classification."""

    def __init__(self, config_path: str):
        super().__init__(
            config_path=config_path,
        )

    @classmethod
    @config_defaults_init("core/model/classification/kolors/mps")
    def from_config(cls, config, **kwargs):
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
