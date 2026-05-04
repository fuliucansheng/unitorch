# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from collections import defaultdict
from typing import List, Optional

import numpy as np


def _compute_average_precision(
    recall: np.ndarray,
    precision: np.ndarray,
) -> float:
    """Compute Average Precision (AP) from a precision-recall curve.

    Uses the area-under-curve method with a monotonically decreasing precision
    envelope (COCO / PASCAL VOC 2010+ style).

    Args:
        recall: Array of recall values.
        precision: Array of precision values.

    Returns:
        AP score as a float.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    change_pts = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_pts + 1] - mrec[change_pts]) * mpre[change_pts + 1]))


def _compute_ap_score(
    predicted_bboxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_classes: List[np.ndarray],
    ground_truth_bboxes: List[np.ndarray],
    ground_truth_classes: List[np.ndarray],
    class_id: Optional[int] = None,
    threshold: float = 0.5,
) -> float:
    """Compute the per-class Average Precision (AP) at a given IoU threshold.

    Args:
        predicted_bboxes: Per-image arrays of predicted boxes ``(N, 4)`` in
                          ``[x1, y1, x2, y2]`` format.
        predicted_scores: Per-image confidence scores for each predicted box.
        predicted_classes: Per-image class IDs for each predicted box.
        ground_truth_bboxes: Per-image ground-truth box arrays ``(M, 4)``.
        ground_truth_classes: Per-image class IDs for each ground-truth box.
        class_id: When set, restrict evaluation to this class; ``None`` evaluates
                  all classes together.
        threshold: IoU threshold above which a detection is a true positive.

    Returns:
        AP score as a float.
    """
    if class_id is not None:
        ground_truth_bboxes = [
            gt_bbox[gt_class == class_id]
            for gt_class, gt_bbox in zip(ground_truth_classes, ground_truth_bboxes)
        ]
        predicted_bboxes = [
            p_bbox[p_class == class_id]
            for p_class, p_bbox in zip(predicted_classes, predicted_bboxes)
        ]
        predicted_scores = [
            p_score[p_class == class_id]
            for p_class, p_score in zip(predicted_classes, predicted_scores)
        ]

    predicted_indexes = [
        np.full(len(predicted_bboxes[i]), i)
        for i in range(len(predicted_bboxes))
    ]

    predicted_bboxes = np.concatenate(predicted_bboxes)
    predicted_scores = np.concatenate(predicted_scores)
    predicted_indexes = np.concatenate(predicted_indexes)
    sort_order = np.argsort(-predicted_scores)

    n_preds = predicted_scores.shape[0]
    true_positives = np.zeros(n_preds)
    false_positives = np.zeros(n_preds)
    gt_matched: defaultdict = defaultdict(set)

    for rank, sort_idx in enumerate(sort_order):
        img_idx = int(predicted_indexes[sort_idx])
        gt_bboxes = ground_truth_bboxes[img_idx]
        pred_bbox = predicted_bboxes[sort_idx]
        best_iou = -np.inf
        best_gt_idx = -1

        if gt_bboxes.size > 0:
            ix1 = np.maximum(gt_bboxes[:, 0], pred_bbox[0])
            iy1 = np.maximum(gt_bboxes[:, 1], pred_bbox[1])
            ix2 = np.minimum(gt_bboxes[:, 2], pred_bbox[2])
            iy2 = np.minimum(gt_bboxes[:, 3], pred_bbox[3])
            inter = np.maximum(ix2 - ix1 + 1.0, 0.0) * np.maximum(iy2 - iy1 + 1.0, 0.0)
            union = (
                (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0)
                + (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.0)
                * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.0)
                - inter
            )
            ious = inter / union
            best_iou = np.max(ious)
            best_gt_idx = int(np.argmax(ious))

        if best_iou > threshold:
            if best_gt_idx not in gt_matched[img_idx]:
                true_positives[rank] = 1
                gt_matched[img_idx].add(best_gt_idx)
            else:
                false_positives[rank] = 1
        else:
            false_positives[rank] = 1

    tp_cum = np.cumsum(true_positives)
    fp_cum = np.cumsum(false_positives)
    n_gt = sum(len(gt) for gt in ground_truth_bboxes)
    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    return _compute_average_precision(recall, precision)


def map_score(
    predicted_bboxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_classes: List[np.ndarray],
    ground_truth_bboxes: List[np.ndarray],
    ground_truth_classes: List[np.ndarray],
) -> float:
    """Compute COCO-style mAP averaged over IoU thresholds 0.50 : 0.05 : 0.95.

    Args:
        predicted_bboxes: Per-image predicted box arrays ``(N, 4)``.
        predicted_scores: Per-image confidence scores.
        predicted_classes: Per-image predicted class IDs.
        ground_truth_bboxes: Per-image ground-truth box arrays ``(M, 4)``.
        ground_truth_classes: Per-image ground-truth class IDs.

    Returns:
        mAP score as a float.
    """
    unique_classes = set(np.concatenate(ground_truth_classes))
    per_threshold_ap = {
        iou: [
            _compute_ap_score(
                predicted_bboxes, predicted_scores, predicted_classes,
                ground_truth_bboxes, ground_truth_classes,
                class_id=cls, threshold=iou / 100,
            )
            for cls in unique_classes
        ]
        for iou in range(50, 100, 5)
    }
    return float(np.mean([np.mean(scores) for scores in per_threshold_ap.values()]))


def map50_score(
    predicted_bboxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_classes: List[np.ndarray],
    ground_truth_bboxes: List[np.ndarray],
    ground_truth_classes: List[np.ndarray],
) -> float:
    """Compute mAP at IoU threshold 0.50 (PASCAL VOC metric).

    Args:
        predicted_bboxes: Per-image predicted box arrays ``(N, 4)``.
        predicted_scores: Per-image confidence scores.
        predicted_classes: Per-image predicted class IDs.
        ground_truth_bboxes: Per-image ground-truth box arrays ``(M, 4)``.
        ground_truth_classes: Per-image ground-truth class IDs.

    Returns:
        mAP\@50 score as a float.
    """
    unique_classes = set(np.concatenate(ground_truth_classes))
    ap_scores = [
        _compute_ap_score(
            predicted_bboxes, predicted_scores, predicted_classes,
            ground_truth_bboxes, ground_truth_classes,
            class_id=cls, threshold=0.5,
        )
        for cls in unique_classes
    ]
    return float(np.mean(ap_scores))
