# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Callable, Optional

import torch
import torch.nn as nn


def sort_by_scores(
    scores: torch.Tensor,
    features_list: list,
    topn: Optional[int] = None,
    shuffle_ties: bool = False,
) -> list:
    """Sort feature tensors by per-example scores.

    Args:
        scores: Shape ``[batch_size, list_size]``.
        features_list: List of tensors with the same shape as *scores*.
        topn: Number of top items to keep; defaults to ``list_size``.
        shuffle_ties: If ``True``, randomly shuffle before sorting to break ties.

    Returns:
        List of tensors sorted by *scores* (descending), each of shape
        ``[batch_size, topn]``.
    """
    assert scores.dim() == 2
    for f in features_list:
        assert f.dim() == 2

    list_size = scores.size(1)
    topn = min(topn, list_size) if topn is not None else list_size

    if shuffle_ties:
        shuffle_ind = torch.argsort(torch.rand_like(scores))
        scores = torch.gather(scores, dim=1, index=shuffle_ind)
        features_list = [torch.gather(f, dim=1, index=shuffle_ind) for f in features_list]

    _, indices = scores.topk(topn, sorted=True)
    return [torch.gather(f, dim=1, index=indices) for f in features_list]


def approx_ranks(logits: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    r"""Compute approximate ranks from logits via a sigmoid approximation.

    The rank of item *i* is approximated as:

    .. math::
        \hat{r}_i = 0.5 + \sum_{j} \sigma(\alpha (s_j - s_i))

    Args:
        logits: Shape ``[batch_size, list_size]``.
        alpha: Sharpness of the sigmoid. Larger values approach hard sorting.

    Returns:
        Approximate ranks with the same shape as *logits*.
    """
    pairs = torch.sigmoid(alpha * (logits.unsqueeze(1) - logits.unsqueeze(2)))
    return pairs.sum(dim=-1) + 0.5


def inverse_max_dcg(
    labels: torch.Tensor,
    gain_fn: Callable = lambda l: torch.pow(2.0, l) - 1.0,
    rank_discount_fn: Callable = lambda r: 1.0 / torch.log1p(r),
    topn: Optional[int] = None,
) -> torch.Tensor:
    """Compute the inverse of the ideal (max) DCG.

    Args:
        labels: Relevance labels of shape ``[batch_size, list_size]``.
        gain_fn: Gain function applied to labels. Default: ``2^label - 1``.
        rank_discount_fn: Discount function applied to ranks. Default: ``1/log(1+rank)``.
        topn: Cutoff rank; defaults to ``list_size``.

    Returns:
        Tensor of shape ``[batch_size, 1]``.
    """
    (ideal_sorted_labels,) = sort_by_scores(labels, [labels], topn=topn)
    rank = (torch.arange(ideal_sorted_labels.size(1)) + 1).float().to(labels.device)
    discounted_gain = (gain_fn(ideal_sorted_labels) * rank_discount_fn(rank)).sum(1, keepdim=True)
    return torch.where(
        discounted_gain > 0.0,
        1.0 / discounted_gain,
        torch.zeros_like(discounted_gain),
    )


def ndcg(
    labels: torch.Tensor,
    ranks: Optional[torch.Tensor] = None,
    perm_mat: Optional[torch.Tensor] = None,
    gain_fn: Callable = lambda l: torch.pow(2.0, l) - 1.0,
    rank_discount_fn: Callable = lambda r: 1.0 / torch.log1p(r),
) -> torch.Tensor:
    """Compute NDCG (or ApproxNDCG / ExpectedNDCG) from labels and ranks.

    Args:
        labels: Relevance labels of shape ``[batch_size, list_size]``.
        ranks: Rank tensor of the same shape as *labels*, or ``None`` (assumes
            labels are already in rank order).
        perm_mat: Permutation matrix of shape ``[batch_size, list_size, list_size]``,
            or ``None``. Mutually exclusive with *ranks*.
        gain_fn: Gain function. Default: ``2^label - 1``.
        rank_discount_fn: Discount function. Default: ``1/log(1+rank)``.

    Returns:
        NDCG tensor of shape ``[batch_size, 1]``.
    """
    if ranks is not None and perm_mat is not None:
        raise ValueError("ranks and perm_mat cannot both be specified.")

    if ranks is None:
        ranks = (torch.arange(labels.size(1)) + 1).to(labels.device)

    discounts = rank_discount_fn(ranks.float())
    gains = gain_fn(labels.float())
    if perm_mat is not None:
        gains = torch.sum(perm_mat * gains.unsqueeze(1), dim=-1)
    dcg = torch.sum(gains * discounts, dim=-1, keepdim=True)
    return dcg * inverse_max_dcg(labels, gain_fn=gain_fn, rank_discount_fn=rank_discount_fn)


class ListMLELoss(nn.Module):
    """ListMLE loss for learning-to-rank."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target * masks
        input = input + (1 - masks) * torch.tensor(1e-6).log()
        scores = torch.where(
            masks.bool(),
            target,
            torch.min(target, dim=1, keepdim=True)[0] - 1e-6,
        )
        sorted_labels, sorted_logits = sort_by_scores(scores, [target, input], shuffle_ties=True)
        sorted_logits = sorted_logits - sorted_logits.max(dim=1, keepdim=True)[0]
        sums = torch.flip(
            torch.cumsum(torch.flip(sorted_logits.exp(), dims=[-1]), dim=1), dims=[-1]
        )
        loss = (sums.log() - sorted_logits).sum(dim=1)

        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class ApproxNDCGLoss(nn.Module):
    """Approx-NDCG loss for learning-to-rank."""

    def __init__(self, reduction: str = "mean", alpha: float = 10.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target * masks
        input = input + (1 - masks) * (input.min(dim=-1, keepdim=True)[0] - 1e3)
        ranks = approx_ranks(input, alpha=self.alpha)
        loss = -ndcg(target, ranks)

        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class ApproxMRRLoss(nn.Module):
    """Approx-MRR loss for learning-to-rank."""

    def __init__(self, reduction: str = "mean", alpha: float = 10.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target * masks
        input = input + (1 - masks) * (input.min(dim=-1, keepdim=True)[0] - 1e3)
        rr = 1.0 / approx_ranks(input, alpha=self.alpha)
        mrr = torch.sum(rr * target, dim=-1, keepdim=True) / torch.sum(
            target, dim=-1, keepdim=True
        )
        loss = -mrr

        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss
