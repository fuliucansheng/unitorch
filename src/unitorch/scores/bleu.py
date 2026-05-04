# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import collections
import math
from typing import List, Optional, Tuple, Union

import torch

from unitorch.scores._utils import convert_tensor_to_strings, remove_strings_ignore_tokens


def _get_ngrams(segment: List, max_order: int) -> collections.Counter:
    """Extract all n-grams up to *max_order* from *segment*.

    Args:
        segment: Tokenised sequence to extract n-grams from.
        max_order: Maximum n-gram order (inclusive).

    Returns:
        A :class:`collections.Counter` mapping each n-gram tuple to its count.
    """
    counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(segment) - order + 1):
            counts[tuple(segment[i : i + order])] += 1
    return counts


def _compute_bleu(
    reference_corpus: List[List[List]],
    translation_corpus: List[List],
    max_order: int = 4,
    smooth: bool = False,
) -> Tuple[float, List[float], float, float, int, int]:
    """Compute the BLEU score for *translation_corpus* against *reference_corpus*.

    Args:
        reference_corpus: For each sentence, a list of reference token lists.
        translation_corpus: List of hypothesis token lists to evaluate.
        max_order: Maximum n-gram order used when computing the score.
        smooth: Apply Lin et al. (2004) smoothing when ``True``.

    Returns:
        A 6-tuple of ``(bleu, precisions, brevity_penalty, ratio,
        translation_length, reference_length)``.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0

    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngrams: collections.Counter = collections.Counter()
        for reference in references:
            merged_ref_ngrams |= _get_ngrams(reference, max_order)
            translation_ngrams = _get_ngrams(translation, max_order)
            overlap = translation_ngrams & merged_ref_ngrams

        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count

        for order in range(1, max_order + 1):
            possible = len(translation) - order + 1
            if possible > 0:
                possible_matches_by_order[order - 1] += possible

    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]

    if min(precisions) > 0:
        geo_mean = math.exp(sum((1.0 / max_order) * math.log(p) for p in precisions))
    else:
        geo_mean = 0.0

    ratio = float(translation_length) / reference_length
    brevity_penalty = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / ratio)
    bleu = geo_mean * brevity_penalty

    return bleu, precisions, brevity_penalty, ratio, translation_length, reference_length


def bleu_score(
    y_true: List[Union[str, int, List[Union[str, int]]]],
    y_pred: List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens: Optional[List[Union[str, int]]] = None,
) -> float:
    """Compute the corpus-level BLEU score.

    Args:
        y_true: Reference sequences. Each element may be a string list, an
                integer list, or (for multiple references) a list of such lists.
                Tensors of shape ``(N, T)`` or ``(N, R, T)`` are also accepted.
        y_pred: Hypothesis sequences in the same format as *y_true*.
        ignore_tokens: Tokens to strip from both references and hypotheses
                       before scoring.

    Returns:
        Corpus-level BLEU score as a float in ``[0, 1]``.
    """
    if isinstance(y_true, torch.Tensor):
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(1)
        y_true = convert_tensor_to_strings(y_true)

    if isinstance(y_pred, torch.Tensor) and y_pred.dim() == 2:
        y_pred = convert_tensor_to_strings(y_pred)

    if ignore_tokens is not None:
        ignore_tokens = [str(t) for t in ignore_tokens]

    y_true = remove_strings_ignore_tokens(y_true, ignore_tokens=ignore_tokens)
    y_pred = remove_strings_ignore_tokens(y_pred, ignore_tokens=ignore_tokens)
    return _compute_bleu(y_true, y_pred)[0]
