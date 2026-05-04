# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import itertools
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from unitorch.scores._utils import convert_tensor_to_strings, remove_strings_ignore_tokens


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_ngrams(n: int, text: List) -> Set[Tuple]:
    """Return the set of all n-grams of order *n* from *text*."""
    return {tuple(text[i : i + n]) for i in range(len(text) - n + 1)}


def _split_into_words(sentences: List[str]) -> List[str]:
    """Flatten a list of whitespace-tokenised sentences into a word list."""
    return list(itertools.chain.from_iterable(s.split() for s in sentences))


def _get_word_ngrams(n: int, sentences: List[str]) -> Set[Tuple]:
    """Return the set of word n-grams of order *n* across *sentences*."""
    assert n > 0 and len(sentences) > 0
    return _get_ngrams(n, _split_into_words(sentences))


def _lcs_table(x: List, y: List) -> Dict[Tuple[int, int], int]:
    """Build the DP table for the Longest Common Subsequence of *x* and *y*.

    Runs in O(|x| · |y|) time and space.
    """
    n, m = len(x), len(y)
    table: Dict[Tuple[int, int], int] = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _len_lcs(x: List, y: List) -> int:
    """Return the length of the Longest Common Subsequence of *x* and *y*."""
    return _lcs_table(x, y)[len(x), len(y)]


def _recon_lcs(x: List, y: List) -> Tuple:
    """Return the Longest Common Subsequence of *x* and *y* as a tuple."""
    table = _lcs_table(x, y)

    def _recon(i: int, j: int) -> List:
        if i == 0 or j == 0:
            return []
        if x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [x[i - 1]]
        if table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        return _recon(i, j - 1)

    return tuple(_recon(len(x), len(y)))


def _f_r_p_rouge_n(
    evaluated_count: int,
    reference_count: int,
    overlapping_count: int,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for a ROUGE-N overlap."""
    precision = overlapping_count / evaluated_count if evaluated_count else 0.0
    recall = overlapping_count / reference_count if reference_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {"f": f1, "p": precision, "r": recall}


def _rouge_n(
    evaluated_sentences: List[str],
    reference_sentences: List[str],
    n: int = 2,
) -> Dict[str, float]:
    """Compute ROUGE-N F1, precision, and recall for a single pair."""
    if not evaluated_sentences:
        raise ValueError("Hypothesis is empty.")
    if not reference_sentences:
        raise ValueError("Reference is empty.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    overlap = len(evaluated_ngrams & reference_ngrams)
    return _f_r_p_rouge_n(len(evaluated_ngrams), len(reference_ngrams), overlap)


def _union_lcs(
    evaluated_sentences: List[str],
    reference_sentence: str,
    prev_union: Optional[Set] = None,
) -> Tuple[int, Set]:
    """Compute the union LCS count between *reference_sentence* and *evaluated_sentences*.

    Returns the number of *new* LCS tokens added beyond *prev_union*, and the
    updated union set.
    """
    if not evaluated_sentences:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = prev_union or set()
    prev_count = len(lcs_union)
    reference_words = _split_into_words([reference_sentence])

    combined_lcs_length = 0
    for sentence in evaluated_sentences:
        lcs = set(_recon_lcs(reference_words, _split_into_words([sentence])))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union | lcs

    return len(lcs_union) - prev_count, lcs_union


def _rouge_l_summary_level(
    evaluated_sentences: List[str],
    reference_sentences: List[str],
) -> Dict[str, float]:
    """Compute summary-level ROUGE-L F1, precision, and recall."""
    if not evaluated_sentences or not reference_sentences:
        raise ValueError("Collections must contain at least 1 sentence.")

    m = len(set(_split_into_words(reference_sentences)))
    n = len(set(_split_into_words(evaluated_sentences)))

    union: Set = set()
    llcs = 0
    for ref_s in reference_sentences:
        lcs_count, union = _union_lcs(evaluated_sentences, ref_s, prev_union=union)
        llcs += lcs_count

    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    f_lcs = (1 + beta ** 2) * r_lcs * p_lcs / (r_lcs + beta ** 2 * p_lcs + 1e-12)
    return {"f": f_lcs, "p": p_lcs, "r": r_lcs}


def _multi_rouge_n(
    sequences: List[List[str]],
    scores_ids: List[Tuple[int, int]],
    n: int = 2,
) -> List[Dict[str, float]]:
    """Compute ROUGE-N for multiple (hypothesis, reference) index pairs efficiently.

    Pre-computes n-gram sets for all sequences so each is processed only once.

    Args:
        sequences: List of token sequences (hypotheses and references combined).
        scores_ids: List of ``(hyp_idx, ref_idx)`` pairs into *sequences*.
        n: N-gram order.

    Returns:
        A list of ``{"f", "p", "r"}`` dicts, one per pair in *scores_ids*.
    """
    ngrams = [_get_word_ngrams(n, seq) for seq in sequences]
    counts = [len(ng) for ng in ngrams]
    return [
        _f_r_p_rouge_n(
            counts[hyp_id],
            counts[ref_id],
            len(ngrams[hyp_id] & ngrams[ref_id]),
        )
        for hyp_id, ref_id in scores_ids
    ]


# ---------------------------------------------------------------------------
# Shared pre-processing
# ---------------------------------------------------------------------------

def _prepare_inputs(
    y_true,
    y_pred,
    ignore_tokens: Optional[List[Union[str, int]]],
):
    """Normalise inputs to string lists and strip ignore tokens."""
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
    # Take the first reference for each example.
    y_true = [refs[0] for refs in y_true]
    return y_true, y_pred


def _average_rouge(rouge_fn, y_true, y_pred) -> Dict[str, float]:
    """Average per-sentence ROUGE scores across a corpus."""
    totals = {"p": 0.0, "r": 0.0, "f": 0.0}
    for ref, hyp in zip(y_true, y_pred):
        scores = rouge_fn(hyp, ref)
        for key in totals:
            totals[key] += scores[key]
    n = len(y_pred)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rouge1_score(
    y_true: List[Union[str, int, List[Union[str, int]]]],
    y_pred: List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens: Optional[List[Union[str, int]]] = None,
) -> Dict[str, float]:
    """Compute corpus-level ROUGE-1 precision, recall, and F1.

    Args:
        y_true: Reference sequences (tokens or nested lists); tensors accepted.
        y_pred: Hypothesis sequences in the same format.
        ignore_tokens: Tokens stripped from both inputs before scoring.

    Returns:
        ``{"pre": float, "rec": float, "f1": float}``
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred, ignore_tokens)
    scores = _average_rouge(lambda h, r: _rouge_n(h, r, n=1), y_true, y_pred)
    return {"pre": scores["p"], "rec": scores["r"], "f1": scores["f"]}


def rouge2_score(
    y_true: List[Union[str, int, List[Union[str, int]]]],
    y_pred: List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens: Optional[List[Union[str, int]]] = None,
) -> Dict[str, float]:
    """Compute corpus-level ROUGE-2 precision, recall, and F1.

    Args:
        y_true: Reference sequences (tokens or nested lists); tensors accepted.
        y_pred: Hypothesis sequences in the same format.
        ignore_tokens: Tokens stripped from both inputs before scoring.

    Returns:
        ``{"pre": float, "rec": float, "f1": float}``
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred, ignore_tokens)
    scores = _average_rouge(lambda h, r: _rouge_n(h, r, n=2), y_true, y_pred)
    return {"pre": scores["p"], "rec": scores["r"], "f1": scores["f"]}


def rougel_score(
    y_true: List[Union[str, int, List[Union[str, int]]]],
    y_pred: List[Union[str, int, List[Union[str, int]]]],
    ignore_tokens: Optional[List[Union[str, int]]] = None,
) -> Dict[str, float]:
    """Compute corpus-level ROUGE-L precision, recall, and F1.

    Args:
        y_true: Reference sequences (tokens or nested lists); tensors accepted.
        y_pred: Hypothesis sequences in the same format.
        ignore_tokens: Tokens stripped from both inputs before scoring.

    Returns:
        ``{"pre": float, "rec": float, "f1": float}``
    """
    y_true, y_pred = _prepare_inputs(y_true, y_pred, ignore_tokens)
    scores = _average_rouge(_rouge_l_summary_level, y_true, y_pred)
    return {"pre": scores["p"], "rec": scores["r"], "f1": scores["f"]}
