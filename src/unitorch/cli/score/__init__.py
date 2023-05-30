# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.score import (
    accuracy_score,
    recall_score,
    f1_score,
    bleu_score,
    map_score,
    map50_score,
    rouge1_score,
    rouge2_score,
    rougel_score,
    roc_auc_score,
    ndcg_score,
    matthews_corrcoef,
    auc,
    precision_recall_curve,
)
from unitorch.cli import add_default_section_for_init, register_score
from unitorch.cli.models import (
    ModelOutputs,
    ModelTargets,
    ClassificationOutputs,
    ClassificationTargets,
    RankingOutputs,
    RankingTargets,
    GenerationOutputs,
    GenerationTargets,
    DetectionOutputs,
    DetectionTargets,
    SegmentationOutputs,
    SegmentationTargets,
    LossOutputs,
)


class Score(nn.Module):
    pass


@register_score("core/score/acc")
class AccuracyScore(Score):
    def __init__(self, gate: Optional[float] = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("core/score/acc")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[ClassificationOutputs, GenerationOutputs, SegmentationOutputs],
        targets: Union[ClassificationTargets, GenerationTargets, SegmentationTargets],
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
            outputs = outputs.view(-1, outputs.size(-1))

        if isinstance(targets, GenerationTargets):
            targets = targets.refs
            targets = targets.view(-1)

        if isinstance(outputs, SegmentationOutputs):
            outputs = outputs.masks
            outputs = torch.cat([t.view(-1, t.size(-1)) for t in outputs])

        if isinstance(targets, SegmentationTargets):
            targets = targets.targets
            targets = torch.cat([t.view(-1) for t in targets])

        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs

        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = (
                outputs.argmax(dim=-1)
                if outputs.size(-1) > 1
                else outputs[:, 0] > self.gate
            )

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return accuracy_score(targets, outputs)


@register_score("core/score/rec")
class RecallScore(Score):
    def __init__(self, gate: Optional[float] = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("core/score/rec")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: ClassificationOutputs,
        targets: ClassificationTargets,
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = (
                outputs.argmax(dim=-1)
                if outputs.size(-1) > 1
                else outputs[:, 0] > self.gate
            )

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return recall_score(targets, outputs, average="micro")


@register_score("core/score/f1")
class F1Score(Score):
    def __init__(self, gate: Optional[float] = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("core/score/f1")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: ClassificationOutputs,
        targets: ClassificationTargets,
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = (
                outputs.argmax(dim=-1)
                if outputs.size(-1) > 1
                else outputs[:, 0] > self.gate
            )

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return f1_score(targets, outputs, average="micro")


@register_score("core/score/auc")
class AUCScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/auc")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: ClassificationOutputs,
        targets: ClassificationTargets,
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = outputs[:, 1] if outputs.size(-1) > 1 else outputs[:, 0]

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return roc_auc_score(targets, outputs)


@register_score("core/score/pr_auc")
class PRAUCScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/pr_auc")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: ClassificationOutputs,
        targets: ClassificationTargets,
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = outputs[:, 1] if outputs.size(-1) > 1 else outputs[:, 0]

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1
        precision, recall, _ = precision_recall_curve(targets, outputs)

        return auc(recall, precision)


@register_score("core/score/ndcg")
class NDCGScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/ndcg")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: RankingOutputs,
        targets: RankingTargets,
    ):
        if isinstance(outputs, RankingOutputs):
            outputs = outputs.outputs
        if isinstance(targets, RankingTargets):
            masks = targets.masks
            targets = targets.targets

        outputs = outputs + (1 - masks) * (
            torch.min(outputs, -1, keepdim=True)[0] - 1e3
        )
        targets = targets * masks
        return ndcg_score(targets, outputs)


@register_score("core/score/mattcorr")
class MattCorrScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/mattcorr")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: ClassificationOutputs,
        targets: ClassificationTargets,
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = (
                outputs.argmax(dim=-1)
                if outputs.size(-1) > 1
                else outputs[:, 0] > self.gate
            )

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return matthews_corrcoef(targets, outputs)


@register_score("core/score/bleu")
class BleuScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/bleu")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs
        return bleu_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )


@register_score("core/score/rouge1")
class Rouge1Score(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rouge1")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs

        return rouge1_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )["f1"]


@register_score("core/score/rouge2")
class Rouge2Score(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rouge2")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs

        return rouge2_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )["f1"]


@register_score("core/score/rougel")
class RougelScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rougel")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets = None,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs

        return rougel_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )["f1"]


@register_score("core/score/loss")
class LossScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/loss")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: LossOutputs,
        targets: ModelTargets,
    ):
        if isinstance(outputs, LossOutputs):
            loss = outputs.loss

        return -float(torch.mean(loss))


@register_score("core/score/mAP")
class MAPScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/mAP")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: DetectionOutputs,
        targets: DetectionTargets,
    ):
        if isinstance(outputs, DetectionOutputs):
            p_bboxes = outputs.bboxes
            p_scores = outputs.scores
            p_classes = outputs.classes
        if isinstance(targets, DetectionTargets):
            gt_bboxes = targets.bboxes
            gt_classes = targets.classes
        return map_score(
            p_bboxes=[t.numpy() for t in p_bboxes],
            p_scores=[t.numpy() for t in p_scores],
            p_classes=[t.numpy() for t in p_classes],
            gt_bboxes=[t.numpy() for t in gt_bboxes],
            gt_classes=[t.numpy() for t in gt_classes],
        )


@register_score("core/score/mAP50")
class MAP50Score(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/mAP50")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: DetectionOutputs,
        targets: DetectionTargets,
    ):
        if isinstance(outputs, DetectionOutputs):
            p_bboxes = outputs.bboxes
            p_scores = outputs.scores
            p_classes = outputs.classes
        if isinstance(targets, DetectionTargets):
            gt_bboxes = targets.bboxes
            gt_classes = targets.classes
        return map50_score(
            p_bboxes=[t.numpy() for t in p_bboxes],
            p_scores=[t.numpy() for t in p_scores],
            p_classes=[t.numpy() for t in p_classes],
            gt_bboxes=[t.numpy() for t in gt_bboxes],
            gt_classes=[t.numpy() for t in gt_classes],
        )
