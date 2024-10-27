# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.detr import DetrConfig
from transformers.models.detr.modeling_detr import (
    DetrModel,
    DetrMLPPredictionHead,
    DetrMaskHeadSmallConv,
    DetrMHAttentionMap,
)
from transformers.loss.loss_for_object_detection import (
    HungarianMatcher as DetrHungarianMatcher,
    ImageLoss as DetrLoss,
)
from unitorch.models import GenericModel, GenericOutputs
from unitorch.utils import image_list_to_tensor


class DetrForDetection(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        config = DetrConfig.from_json_file(config_path)

        self.model = DetrModel(config)
        if num_classes is not None:
            config.num_labels = num_classes
        self.class_labels_classifier = nn.Linear(config.d_model, config.num_labels + 1)
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3,
        )
        self.config = config
        self.init_weights()

        # modules for loss
        self.enable_auxiliary_loss = config.auxiliary_loss
        matcher = DetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
        )
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = DetrLoss(
            matcher=matcher,
            num_classes=config.num_labels,
            eos_coef=config.eos_coefficient,
            losses=losses,
        )
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
        }
        if self.enable_auxiliary_loss:
            aux_weight_dict = {}
            for i in range(config.decoder_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(aux_weight_dict)

    @property
    def dtype(self):
        """
        `torch.dtype`: which dtype the parameters are (assuming that all the parameters are the same dtype).
        """

        return next(self.parameters()).dtype

    @property
    def device(self):
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        """
        return next(self.parameters()).device

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def forward(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ):
        if not isinstance(images, torch.Tensor):
            images = image_list_to_tensor(images)
        assert images.dim() == 4

        outputs = self.model(images.to(self.dtype))
        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        outputs_loss = {}
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if self.enable_auxiliary_loss:
            intermediate = outputs[4]
            outputs_class = self.class_labels_classifier(intermediate)
            outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            auxiliary_outputs = self._set_aux_loss(
                outputs_class,
                outputs_coord,
            )
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        xyxy2xywh = lambda x: torch.stack(
            [
                (x[..., 0] + x[..., 2]) / 2,
                (x[..., 1] + x[..., 3]) / 2,
                x[..., 2] - x[..., 0],
                x[..., 3] - x[..., 1],
            ],
            -1,
        )
        bboxes = [xyxy2xywh(bbox) for bbox in bboxes]
        labels = [{"class_labels": c, "boxes": b} for b, c in zip(bboxes, classes)]
        loss_dict = self.criterion(outputs_loss, labels)
        loss = sum(
            loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        )
        return loss

    @torch.no_grad()
    def detect(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        norm_bboxes: Optional[bool] = False,
        threshold: Optional[float] = 0.5,
    ):
        image_sizes = [(img.size(-2), img.size(-1)) for img in images]
        if not isinstance(images, torch.Tensor):
            images = image_list_to_tensor(images)
        assert images.dim() == 4

        outputs = self.model(images.to(self.dtype))
        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        logits = logits.softmax(dim=-1)

        scores, classes = list(zip(*[p.max(-1) for p in logits]))
        xywh2xyxy = lambda x: torch.stack(
            [
                x[..., 0] - x[..., 2] / 2,
                x[..., 1] - x[..., 3] / 2,
                x[..., 0] + x[..., 2] / 2,
                x[..., 1] + x[..., 3] / 2,
            ],
            -1,
        )
        pred_boxes = [xywh2xyxy(bbox) for bbox in pred_boxes]
        if not norm_bboxes:
            sizes = image_sizes
            bboxes = [
                b * torch.tensor([s[1], s[0], s[1], s[0]]).to(b)
                for b, s in zip(pred_boxes, sizes)
            ]
        else:
            bboxes = pred_boxes

        bboxes, scores, classes = list(
            zip(
                *[
                    (
                        b[(c != self.config.num_labels) & (s > threshold)],
                        s[(c != self.config.num_labels) & (s > threshold)],
                        c[(c != self.config.num_labels) & (s > threshold)],
                    )
                    for b, s, c in zip(bboxes, scores, classes)
                ]
            )
        )

        outputs = dict(
            {
                "bboxes": list(bboxes),
                "scores": list(scores),
                "classes": list(classes),
            }
        )
        return GenericOutputs(outputs)
