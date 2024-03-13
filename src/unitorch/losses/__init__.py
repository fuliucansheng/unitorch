# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.losses.ranking import ListMLELoss, ApproxNDCGLoss, ApproxMRRLoss


class CELoss(nn.Module):
    def __init__(
        self,
        smoothing_alpha: Optional[float] = 0.0,
        weight: Optional[torch.Tensor] = None,
        reduction: Optional[str] = "mean",
    ):
        super().__init__()
        self.smoothing_alpha = smoothing_alpha
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        assert input.dim() == 2 and target.dim() == 1

        target = target.long()

        if self.smoothing_alpha > 0:
            batch_size, num_classes = input.size()
            smooth_label = torch.full(
                size=(batch_size, num_classes),
                fill_value=self.smoothing_alpha / (num_classes - 1),
            ).to(loss)
            smooth_label.scatter_(
                dim=1,
                index=torch.unsqueeze(target, dim=1),
                value=(1 - self.smoothing_alpha),
            )
            log_logits = torch.nn.functional.log_softmax(input, dim=1)
            loss = torch.sum(log_logits * smooth_label, dim=1)
        else:
            loss = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(
                input, target
            ).squeeze()

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class BCELoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: Optional[str] = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        if target.dim() == 1:
            target = torch.unsqueeze(target, dim=1)
        assert input.dim() == 2 and target.dim() == 2

        target = target.float()
        loss = nn.BCEWithLogitsLoss(weight=self.weight, reduction="none")(input, target)

        loss = torch.sum(loss, dim=1)

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class LMLoss(nn.Module):
    def __init__(
        self,
        reduction: Optional[str] = "mean",
    ):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        assert input.dim() == 3 and target.dim() == 2

        batch_size, seq_len, num_classes = input.size()
        input = input.contiguous().view(batch_size * seq_len, num_classes)
        target = target.contiguous().view(-1)
        target = target.long()
        if masks is None:
            masks = torch.ones_like(target).to(target)
        masks = masks.contiguous().view(-1)
        loss = nn.CrossEntropyLoss(reduction="none")(input, target)
        loss = loss * masks.float()
        loss = loss.contiguous().view(batch_size, seq_len).sum(1) / torch.max(
            masks.contiguous().view(batch_size, seq_len).float().sum(1),
            torch.ones(batch_size).to(masks.device),
        )

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class MSELoss(nn.Module):
    def __init__(
        self,
        reduction: Optional[str] = "mean",
    ):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        assert input.size(0) == target.size(0) and input.numel() == target.numel()
        batch_size = input.size(0)
        loss = nn.MSELoss(reduction="none")(
            input.view(batch_size, -1), target.view(batch_size, -1)
        )

        if loss.dim() > 1:
            loss = torch.sum(loss, dim=1)

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss
