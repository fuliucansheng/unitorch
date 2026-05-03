# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from unitorch.cli import (
    config_defaults_init,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models.modeling_utils import TensorOutputs, TensorTargets


@dataclass
class GenerationOutputs(TensorOutputs, WriterMixin):
    """Outputs for generation models."""

    sequences: torch.Tensor
    sequences_scores: Optional[torch.Tensor] = torch.empty(0)


@dataclass
class GenerationTargets(TensorTargets):
    """Targets for generation models."""

    refs: torch.Tensor
    masks: Optional[torch.Tensor] = torch.empty(0)
    sample_weight: Optional[torch.Tensor] = torch.empty(0)


class GenerationProcessor:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    @classmethod
    @config_defaults_init("core/process/generation")
    def from_config(cls, config, **kwargs):
        pass

    @register_process("core/postprocess/generation")
    def _generation(
        self,
        outputs: GenerationOutputs,
    ):
        return WriterOutputs()


def generation_model_decorator(cls):
    class GenerationModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__generation_model__" in kwargs:
                self.model = kwargs.pop("__generation_model__")
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

            assert hasattr(self.model, "generate")

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.generate(*args, **kwargs)

        @classmethod
        def from_config(_cls, cfg, **kwargs):
            model = cls.from_config(cfg, **kwargs)
            return _cls(__generation_model__=model)

    return GenerationModel
