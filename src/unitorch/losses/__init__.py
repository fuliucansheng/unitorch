# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional

import torch
import torch.nn as nn
from unitorch.losses.ranking import ApproxMRRLoss, ApproxNDCGLoss, ListMLELoss


class CELoss(nn.Module):
    """Cross-entropy loss with optional label smoothing and per-sample weighting."""

    def __init__(
        self,
        smoothing_alpha: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing_alpha = smoothing_alpha
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input.dim() == 2 and target.dim() == 1
        target = target.long()

        if self.smoothing_alpha > 0:
            batch_size, num_classes = input.size()
            smooth_label = torch.full(
                (batch_size, num_classes),
                fill_value=self.smoothing_alpha / (num_classes - 1),
                device=input.device,
                dtype=input.dtype,
            )
            smooth_label.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing_alpha)
            loss = -torch.sum(torch.log_softmax(input, dim=1) * smooth_label, dim=1)
        else:
            loss = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(input, target).squeeze()

        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class BCELoss(nn.Module):
    """Binary cross-entropy loss with optional per-sample weighting."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(1)
        assert input.dim() == 2 and target.dim() == 2
        loss = nn.BCEWithLogitsLoss(weight=self.weight, reduction="none")(input, target.float())
        loss = loss.sum(dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class LMLoss(nn.Module):
    """Language-modelling cross-entropy loss with mask support."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input.dim() == 3 and target.dim() == 2
        batch_size, seq_len, num_classes = input.size()
        input = input.contiguous().view(batch_size * seq_len, num_classes)
        target = target.contiguous().view(-1).long()

        if masks is None:
            masks = torch.ones_like(target)
        masks = masks.contiguous().view(-1)

        loss = nn.CrossEntropyLoss(reduction="none")(input, target) * masks.float()
        token_counts = masks.view(batch_size, seq_len).float().sum(1).clamp(min=1.0)
        loss = loss.view(batch_size, seq_len).sum(1) / token_counts

        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class MSELoss(nn.Module):
    """Mean-squared-error loss with optional per-sample weighting."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input.size(0) == target.size(0) and input.numel() == target.numel()
        batch_size = input.size(0)
        loss = nn.MSELoss(reduction="none")(input.view(batch_size, -1), target.view(batch_size, -1))
        if loss.dim() > 1:
            loss = loss.sum(dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss
