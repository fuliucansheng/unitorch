# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.losses import (
    ListMLELoss as _ListMLELoss,
    ApproxNDCGLoss as _ApproxNDCGLoss,
    ApproxMRRLoss as _ApproxMRRLoss,
)
from unitorch.cli import add_default_section_for_init, register_loss
from unitorch.cli.models import (
    RankingOutputs,
    RankingTargets,
)


@register_loss("core/loss/ranking/listmle")
class ListMLELoss(_ListMLELoss):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/loss/ranking/listmle")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: RankingOutputs,
        targets: RankingTargets,
    ):
        outputs = outputs.outputs
        masks = targets.masks
        weights = targets.sample_weight
        targets = targets.targets

        return super().forward(
            input=outputs,
            target=targets,
            masks=masks if masks.numel() > 0 else None,
            sample_weight=weights if weights.numel() > 0 else None,
        )


@register_loss("core/loss/ranking/approxndcg")
class ApproxNDCGLoss(_ApproxNDCGLoss):
    def __init__(
        self,
        alpha: Optional[float] = 10.0,
    ):
        super().__init__(
            alpha=alpha,
        )

    @classmethod
    @add_default_section_for_init("core/loss/ranking/approxndcg")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: RankingOutputs,
        targets: RankingTargets,
    ):
        outputs = outputs.outputs
        masks = targets.masks
        weights = targets.sample_weight
        targets = targets.targets

        return super().forward(
            input=outputs,
            target=targets,
            masks=masks if masks.numel() > 0 else None,
            sample_weight=weights if weights.numel() > 0 else None,
        )


@register_loss("core/loss/ranking/approxmrr")
class ApproxMRRLoss(_ApproxMRRLoss):
    def __init__(
        self,
        alpha: Optional[float] = 0.0,
    ):
        super().__init__(
            alpha=alpha,
        )

    @classmethod
    @add_default_section_for_init("core/loss/ranking/approxmrr")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: RankingOutputs,
        targets: RankingTargets,
    ):
        outputs = outputs.outputs
        masks = targets.masks
        weights = targets.sample_weight
        targets = targets.targets

        return super().forward(
            input=outputs,
            target=targets,
            masks=masks if masks.numel() > 0 else None,
            sample_weight=weights if weights.numel() > 0 else None,
        )
