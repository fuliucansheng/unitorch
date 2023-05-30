# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class ProphetnetLoss(nn.Module):
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
        assert input.dim() == 4 and target.dim() == 2

        batch_size, ngram, seq_len, num_classes = input.size()
        input = input.contiguous().view(batch_size * ngram * seq_len, num_classes)
        target = target.repeat(1, ngram).contiguous().view(-1)
        target = target.long()
        if masks is None or masks.numel() == 0:
            masks = torch.ones(batch_size, seq_len).to(target)
        masks = masks.repeat(1, ngram).contiguous().view(-1)

        loss = nn.CrossEntropyLoss(reduction="none")(input, target)
        loss = loss * masks.float()
        loss = loss.contiguous().view(batch_size, ngram * seq_len).sum(1) / torch.max(
            masks.contiguous().view(batch_size, ngram * seq_len).float().sum(1),
            torch.ones(batch_size).to(masks.device),
        )

        if sample_weight is not None and sample_weight.numel() > 0:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss
