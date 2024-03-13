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
class DetectionOutputs(ListTensorsOutputs, WriterMixin):
    bboxes: Union[torch.Tensor, List[torch.Tensor]]
    scores: Union[torch.Tensor, List[torch.Tensor]]
    classes: Union[torch.Tensor, List[torch.Tensor]]
    features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = torch.empty(0)


@dataclass
class DetectionTargets(ListTensorsTargets):
    bboxes: Union[torch.Tensor, List[torch.Tensor]]
    classes: Union[torch.Tensor, List[torch.Tensor]]
    sample_weight: Optional[torch.Tensor] = torch.empty(0)


class DetectionProcessor:
    def __init__(
        self,
    ):
        pass

    @classmethod
    @add_default_section_for_init("core/process/detection")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("core/postprocess/detection")
    def _detection(
        self,
        outputs: DetectionOutputs,
    ):
        results = outputs.to_pandas()
        results["bboxes"] = [b.tolist() for b in outputs.bboxes]
        results["scores"] = [s.tolist() for s in outputs.scores]
        results["classes"] = [c.tolist() for c in outputs.classes]
        return WriterOutputs(results)


def detection_model_decorator(cls):
    class DetectionModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__detection_model__" in kwargs:
                self.model = kwargs.pop("__detection_model__")
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

            assert hasattr(self.model, "detect")

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.detect(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__detection_model__=model)

    return DetectionModel
