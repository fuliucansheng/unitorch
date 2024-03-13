# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def _compute_average_precision(
    recall: np.ndarray,
    precision: np.ndarray,
    use_07_metric: Optional[bool] = False,
) -> float:
    """
    Compute Average Precision (AP) given precision and recall.

    Args:
        recall: Array of recall values.
        precision: Array of precision values.
        use_07_metric: Whether to use the VOC 2007 11-point metric.

    Returns:
        AP score.

    """
    if use_07_metric:
        # 11-point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * precision
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _compute_ap_score(
    predicted_bboxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_classes: List[np.ndarray],
    ground_truth_bboxes: List[np.ndarray],
    ground_truth_classes: List[np.ndarray],
    class_id: Optional[int] = None,
    threshold: Optional[float] = 0.5,
) -> float:
    """
    Compute Average Precision (AP) score for a specific class.

    Args:
        predicted_bboxes: List of predicted bounding boxes.
        predicted_scores: List of predicted scores for each bounding box.
        predicted_classes: List of predicted class IDs for each bounding box.
        ground_truth_bboxes: List of ground truth bounding boxes.
        ground_truth_classes: List of true class IDs for each ground truth bounding box.
        class_id: The class ID to compute the AP score.
        threshold: The threshold to determine true positives.

    Returns:
        AP score.

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
            np.array([i] * len(predicted_bboxes[i]))
            for i in range(len(predicted_bboxes))
        ]

    predicted_bboxes = np.concatenate(predicted_bboxes)
    predicted_scores = np.concatenate(predicted_scores)
    predicted_indexes = np.concatenate(predicted_indexes)
    predicted_sort_indexes = np.argsort(-predicted_scores)

    true_positives = np.zeros(predicted_scores.shape[0])
    false_positives = np.zeros(predicted_scores.shape[0])
    gt_bbox_status = defaultdict(set)
    for idx, predicted_sort_index in enumerate(predicted_sort_indexes):
        predicted_index = int(predicted_indexes[predicted_sort_index])
        gt_bboxes = ground_truth_bboxes[predicted_index]
        predicted_bbox = predicted_bboxes[predicted_sort_index]
        max_overlap = -float("inf")
        max_overlap_index = -1
        if gt_bboxes.size > 0:
            ixmin = np.maximum(gt_bboxes[:, 0], predicted_bbox[0])
            iymin = np.maximum(gt_bboxes[:, 1], predicted_bbox[1])
            ixmax = np.minimum(gt_bboxes[:, 2], predicted_bbox[2])
            iymax = np.minimum(gt_bboxes[:, 3], predicted_bbox[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih
            uni = (
                (predicted_bbox[2] - predicted_bbox[0] + 1.0)
                * (predicted_bbox[3] - predicted_bbox[1] + 1.0)
                + (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.0)
                * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.0)
                - inters
            )
            overlaps = inters / uni
            max_overlap = np.max(overlaps)
            max_overlap_index = np.argmax(overlaps)
        if max_overlap > threshold:
            if max_overlap_index not in gt_bbox_status[predicted_index]:
                true_positives[idx] = 1
                gt_bbox_status[predicted_index].add(max_overlap_index)
            else:
                false_positives[idx] = 1
        else:
            false_positives[idx] = 1
    false_positives = np.cumsum(false_positives, axis=0)
    true_positives = np.cumsum(true_positives, axis=0)
    recall = true_positives / float(sum([len(gt) for gt in ground_truth_bboxes]))
    precision = true_positives / np.maximum(
        true_positives + false_positives, np.finfo(np.float64).eps
    )
    ap = _compute_average_precision(recall, precision)
    return ap


def map_score(
    predicted_bboxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_classes: List[np.ndarray],
    ground_truth_bboxes: List[np.ndarray],
    ground_truth_classes: List[np.ndarray],
) -> float:
    """
    Compute Mean Average Precision (mAP) score for all classes.

    Args:
        predicted_bboxes: List of predicted bounding boxes.
        predicted_scores: List of predicted scores for each bounding box.
        predicted_classes: List of predicted class IDs for each bounding box.
        ground_truth_bboxes: List of ground truth bounding boxes.
        ground_truth_classes: List of true class IDs for each ground truth bounding box.

    Returns:
        mAP score.

    """
    unique_classes = set(list(np.concatenate(ground_truth_classes)))
    ap_scores = dict()
    for threshold in range(50, 100, 5):
        ap_scores[threshold] = [
            _compute_ap_score(
                predicted_bboxes,
                predicted_scores,
                predicted_classes,
                ground_truth_bboxes,
                ground_truth_classes,
                class_id,
                threshold / 100,
            )
            for class_id in unique_classes
        ]

    mAP = {iou: np.mean(scores) for iou, scores in ap_scores.items()}
    return np.mean(list(mAP.values()))


def map50_score(
    predicted_bboxes: List[np.ndarray],
    predicted_scores: List[np.ndarray],
    predicted_classes: List[np.ndarray],
    ground_truth_bboxes: List[np.ndarray],
    ground_truth_classes: List[np.ndarray],
) -> float:
    """
    Compute Mean Average Precision (mAP@50) score for all classes.

    Args:
        predicted_bboxes: List of predicted bounding boxes.
        predicted_scores: List of predicted scores for each bounding box.
        predicted_classes: List of predicted class IDs for each bounding box.
        ground_truth_bboxes: List of ground truth bounding boxes.
        ground_truth_classes: List of true class IDs for each ground truth bounding box.

    Returns:
        mAP@50 score.

    """
    unique_classes = set(list(np.concatenate(ground_truth_classes)))
    ap_scores = [
        _compute_ap_score(
            predicted_bboxes,
            predicted_scores,
            predicted_classes,
            ground_truth_bboxes,
            ground_truth_classes,
            class_id,
            0.5,
        )
        for class_id in unique_classes
    ]

    return np.mean(ap_scores)
