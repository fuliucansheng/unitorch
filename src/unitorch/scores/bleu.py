# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import collections
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.scores import convert_tensor_to_strings, remove_strings_ignore_tokens


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
            translation_ngram_counts = _get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (
                possible_matches_by_order[i] + 1.0
            )
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (
                    float(matches_by_order[i]) / possible_matches_by_order[i]
                )
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def bleu_score(
    y_true: List[Union[str, int, List[Union[str, int]]]],
    y_pred: List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens: Optional[List[Union[str, int]]] = None,
):
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
