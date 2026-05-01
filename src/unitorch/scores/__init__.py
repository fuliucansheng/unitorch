# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from unitorch.scores._utils import convert_tensor_to_strings, remove_strings_ignore_tokens
from unitorch.scores.bleu import bleu_score
from unitorch.scores.map import map50_score, map_score
from unitorch.scores.rouge import rouge1_score, rouge2_score, rougel_score


def pearsonr(y_true, y_pred) -> float:
    """Return the Pearson correlation coefficient between *y_true* and *y_pred*."""
    return np.corrcoef(y_true, y_pred)[0, 1]
