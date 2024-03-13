# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def sort_by_scores(
    scores,
    features_list,
    topn=None,
    shuffle_ties=False,
):
    """Sorts example features according to per-example scores.
    Args:
      scores: A `Tensor` of shape [batch_size, list_size] representing the
        per-example scores.
      features_list: A list of `Tensor`s with the same shape as scores to be
        sorted.
      topn: An integer as the cutoff of examples in the sorted list.
      shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    Returns:
      A list of `Tensor`s as the list of sorted features by `scores`.
    """
    score_size = list(scores.size())
    assert len(score_size) == 2
    for f in features_list:
        assert len(f.size()) == 2
    list_size = score_size[1]
    if topn is None:
        topn = list_size
    topn = min(topn, list_size)
    if shuffle_ties:
        shuffle_ind = torch.argsort(torch.rand(score_size)).to(scores.device)
        scores = torch.gather(scores, dim=1, index=shuffle_ind)
        features_list = [
            torch.gather(f, dim=1, index=shuffle_ind) for f in features_list
        ]
    _, indices = scores.topk(topn, sorted=True)
    return [torch.gather(f, dim=1, index=indices) for f in features_list]


def approx_ranks(logits, alpha=10.0):
    r"""Computes approximate ranks given a list of logits.

    Given a list of logits, the rank of an item in the list is simply
    one plus the total number of items with a larger logit. In other words,

      rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},

    where "I" is the indicator function. The indicator function can be
    approximated by a generalized sigmoid:

      I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).

    This function approximates the rank of an item using this sigmoid
    approximation to the indicator function. This technique is at the core
    of "A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.

    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      alpha: Exponent of the generalized sigmoid function.

    Returns:
      A `Tensor` of ranks with the same shape as logits.
    """
    """
    list_size=tf.shape(input=logits)[1]
    x = tf.tile(tf.expand_dims(logits, 2), [1, 1, list_size])
    y = tf.tile(tf.expand_dims(logits, 1), [1, list_size, 1])
    pairs = tf.sigmoid(alpha * (y - x))
    return tf.reduce_sum(input_tensor=pairs, axis=-1) + .5
    """
    x = torch.unsqueeze(logits, 2)
    y = torch.unsqueeze(logits, 1)
    pairs = torch.sigmoid(alpha * (y - x))
    return torch.sum(pairs, -1) + 0.5


def inverse_max_dcg(
    labels,
    gain_fn=lambda labels: torch.pow(2.0, labels) - 1.0,
    rank_discount_fn=lambda rank: 1.0 / torch.log1p(rank),
    topn=None,
):
    """Computes the inverse of max DCG.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size]. Each value is the
        graded relevance of the corresponding item.
      gain_fn: A gain function. By default this is set to: 2^label - 1.
      rank_discount_fn: A discount function. By default this is set to:
        1/log(1+rank).
      topn: An integer as the cutoff of examples in the sorted list.

    Returns:
      A `Tensor` with shape [batch_size, 1].
    """
    (ideal_sorted_labels,) = sort_by_scores(labels, [labels], topn=topn)
    rank = torch.arange(ideal_sorted_labels.size(1)) + 1
    discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(
        rank.float().to(ideal_sorted_labels.device)
    )
    discounted_gain = torch.sum(discounted_gain, 1, keepdim=True)
    return torch.where(
        torch.gt(discounted_gain, 0.0),
        1.0 / discounted_gain,
        torch.zeros_like(discounted_gain),
    )


def ndcg(
    labels,
    ranks=None,
    perm_mat=None,
    gain_fn=lambda labels: torch.pow(2.0, labels) - 1.0,
    rank_discount_fn=lambda rank: 1.0 / torch.log1p(rank),
):
    """Computes NDCG from labels and ranks.

    Args:
      labels: A `Tensor` with shape [batch_size, list_size], representing graded
        relevance.
      ranks: A `Tensor` of the same shape as labels, or [1, list_size], or None.
        If ranks=None, we assume the labels are sorted in their rank.
      perm_mat: A `Tensor` with shape [batch_size, list_size, list_size] or None.
        Permutation matrices with rows correpond to the ranks and columns
        correspond to the indices. An argmax over each row gives the index of the
        element at the corresponding rank.

    Returns:
      A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
    """
    if ranks is not None and perm_mat is not None:
        raise ValueError("Cannot use both ranks and perm_mat simultaneously.")

    if ranks is None:
        list_size = labels.size(1)
        ranks = torch.arange(list_size) + 1
    discounts = rank_discount_fn(ranks.float())
    gains = gain_fn(labels.float())
    if perm_mat is not None:
        gains = torch.sum(perm_mat * torch.unsqueeze(gains, 1), -1)
    dcg = torch.sum(gains * discounts, -1, keepdim=True)
    return dcg * inverse_max_dcg(
        labels,
        gain_fn=gain_fn,
        rank_discount_fn=rank_discount_fn,
    )


class ListMLELoss(nn.Module):
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
        masks: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        target = target * masks
        input = input + (1 - masks) * torch.tensor(1e-6).log()
        scores = torch.where(
            masks.bool(),
            target,
            torch.min(target, axis=1, keepdims=True)[0] - 1e-6,
        )
        sorted_labels, sorted_logits = sort_by_scores(
            scores,
            [target, input],
            shuffle_ties=True,
        )
        raw_max = torch.max(sorted_logits, dim=1, keepdim=True)[0]
        sorted_logits = sorted_logits - raw_max

        sums = torch.flip(
            torch.cumsum(torch.flip(torch.exp(sorted_logits), [-1]), dim=1),
            [-1],
        )

        sums = torch.log(sums) - sorted_logits
        loss = torch.sum(sums, dim=1)

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class ApproxNDCGLoss(nn.Module):
    def __init__(
        self,
        reduction: Optional[str] = "mean",
        alpha: Optional[float] = 10.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        target = target * masks
        input = input + (1 - masks) * (torch.min(input, -1, keepdim=True)[0] - 1e3)
        ranks = approx_ranks(input, alpha=self.alpha)
        loss = -ndcg(target, ranks)

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss


class ApproxMRRLoss(nn.Module):
    def __init__(
        self,
        reduction: Optional[str] = "mean",
        alpha: Optional[float] = 10.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        target = target * masks
        input = input + (1 - masks) * (torch.min(input, -1, keepdim=True)[0] - 1e3)
        rr = 1.0 / approx_ranks(input, alpha=self.alpha)
        rr = torch.sum(input * target, -1, keepdim=True)
        mrr = rr / torch.sum(target, -1, keepdim=True)
        loss = -mrr

        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == "mean":
            loss = torch.mean(loss)

        return loss
