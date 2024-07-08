# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.losses import (
    CELoss as _CELoss,
    BCELoss as _BCELoss,
    LMLoss as _LMLoss,
    MSELoss as _MSELoss,
)
from unitorch.cli import add_default_section_for_init, register_loss
from unitorch.cli.models import (
    ClassificationOutputs,
    ClassificationTargets,
    SegmentationOutputs,
    SegmentationTargets,
    GenerationOutputs,
    GenerationTargets,
)
import unitorch.cli.losses.ranking


@register_loss("core/loss/ce")
class CELoss(_CELoss):
    def __init__(
        self,
        smoothing_alpha: Optional[float] = 0.0,
    ):
        super().__init__(
            smoothing_alpha,
        )

    @classmethod
    @add_default_section_for_init("core/loss/ce")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[ClassificationOutputs, SegmentationOutputs],
        targets: Union[ClassificationTargets, SegmentationTargets],
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            weights = targets.sample_weight
            targets = targets.targets

        if isinstance(outputs, SegmentationOutputs):
            outputs = torch.cat([m.view(-1, m.size(-1)) for m in outputs.masks], dim=0)
        if isinstance(targets, SegmentationTargets):
            weights = torch.empty(0)
            targets = torch.cat([m.view(-1) for m in targets.targets], dim=0)

        return super().forward(
            input=outputs,
            target=targets,
            sample_weight=weights if weights.numel() > 0 else None,
        )


@register_loss("core/loss/bce")
class BCELoss(_BCELoss):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/loss/bce")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[ClassificationOutputs, SegmentationOutputs],
        targets: Union[ClassificationTargets, SegmentationTargets],
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            weights = targets.sample_weight
            targets = targets.targets

        if isinstance(outputs, SegmentationOutputs):
            outputs = torch.cat([m.view(-1, m.size(-1)) for m in outputs.masks], dim=0)
        if isinstance(targets, SegmentationTargets):
            weights = torch.empty(0)
            targets = torch.cat([m.view(-1) for m in targets.targets], dim=0)

        return super().forward(
            input=outputs,
            target=targets,
            sample_weight=weights if weights.numel() > 0 else None,
        )


@register_loss("core/loss/lm")
class LMLoss(_LMLoss):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/loss/lm")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        outputs = outputs.sequences
        masks = targets.masks
        weights = targets.sample_weight
        targets = targets.refs

        return super().forward(
            input=outputs,
            target=targets,
            masks=masks if masks.numel() > 0 else None,
            sample_weight=weights if weights.numel() > 0 else None,
        )


@register_loss("core/loss/mse")
class MSELoss(_MSELoss):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/loss/mse")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[ClassificationOutputs, SegmentationOutputs],
        targets: Union[ClassificationTargets, SegmentationTargets],
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            weights = targets.sample_weight
            targets = targets.targets

        if isinstance(outputs, SegmentationOutputs):
            outputs = torch.cat([m.view(-1) for m in outputs.masks], dim=0)
        if isinstance(targets, SegmentationTargets):
            weights = torch.empty(0)
            targets = torch.cat([m.view(-1) for m in targets.targets], dim=0)

        return super().forward(
            input=outputs,
            target=targets,
            sample_weight=weights if weights.numel() > 0 else None,
        )
