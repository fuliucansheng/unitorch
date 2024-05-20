# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    ndcg_score,
    auc,
    precision_recall_curve,
)
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def convert_tensor_to_strings(inputs: torch.Tensor) -> Any:
    if inputs.dim() == 0:
        return str(inputs.item())
    elif inputs.dim() == 1:
        return [str(element.item()) for element in inputs]
    else:
        return [convert_tensor_to_strings(tensor) for tensor in inputs]


def remove_strings_ignore_tokens(inputs: Any, ignore_tokens: Optional[Set[str]]) -> Any:
    if ignore_tokens is None:
        return inputs

    if isinstance(inputs, list):
        return [
            remove_strings_ignore_tokens(element, ignore_tokens) for element in inputs
        ]
    elif isinstance(inputs, dict):
        return {
            key: remove_strings_ignore_tokens(value, ignore_tokens)
            for key, value in inputs.items()
        }
    elif isinstance(inputs, tuple):
        return tuple(
            remove_strings_ignore_tokens(element, ignore_tokens) for element in inputs
        )
    elif isinstance(inputs, set):
        return {
            remove_strings_ignore_tokens(element, ignore_tokens) for element in inputs
        }
    elif isinstance(inputs, frozenset):
        return frozenset(
            remove_strings_ignore_tokens(element, ignore_tokens) for element in inputs
        )
    else:
        return inputs


pearsonr = lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0, 1]

from unitorch.scores.bleu import bleu_score
from unitorch.scores.rouge import rouge1_score, rouge2_score, rougel_score
from unitorch.scores.map import map_score, map50_score
