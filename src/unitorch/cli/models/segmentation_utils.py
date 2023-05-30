# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models.modeling_utils import ListTensorsOutputs, ListTensorsTargets


@dataclass
class SegmentationOutputs(ListTensorsOutputs, WriterMixin):
    outputs: List[torch.Tensor]


@dataclass
class SegmentationTargets(ListTensorsTargets):
    targets: List[torch.Tensor]
    sample_weight: Optional[torch.Tensor] = torch.empty(0)


class SegmentationProcessor:
    def __init__(
        self,
    ):
        pass

    @classmethod
    @add_default_section_for_init("core/process/segmentation")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("core/postprocess/segmentation")
    def _segmentation(
        self,
        outputs: SegmentationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == len(outputs.outputs)
        results["pixel_class"] = [
            m.numpy().argmax(-1).reshape(-1).tolist() for m in outputs.outputs
        ]
        return WriterOutputs(results)


def segmentation_model_decorator(cls):
    class SegmentationModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__segmentation_model__" in kwargs:
                self.model = kwargs.pop("__segmentation_model__")
            else:
                self.model = cls(*args, **kwargs)

            __more_attrs__ = [
                "load_state_dict",
                "state_dict",
                "save_checkpoint",
                "from_checkpoint",
                "from_pretrained",
            ]
            for __more_attr__ in __more_attrs__:
                setattr(self, __more_attr__, getattr(self.model, __more_attr__))

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.segment(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__segmentation_model__=model)

    return SegmentationModel
