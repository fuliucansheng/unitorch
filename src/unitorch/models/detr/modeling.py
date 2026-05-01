# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers.loss.loss_for_object_detection import HungarianMatcher as DetrHungarianMatcher
from transformers.loss.loss_for_object_detection import ImageLoss as DetrLoss
from transformers.models.detr import DetrConfig
from transformers.models.detr.modeling_detr import DetrMLPPredictionHead, DetrModel

from unitorch.models import GenericModel, GenericOutputs
from unitorch.utils import image_list_to_tensor


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] boxes to [cx, cy, w, h] format."""
    return torch.stack([
        (boxes[..., 0] + boxes[..., 2]) / 2,
        (boxes[..., 1] + boxes[..., 3]) / 2,
        boxes[..., 2] - boxes[..., 0],
        boxes[..., 3] - boxes[..., 1],
    ], dim=-1)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] boxes to [x1, y1, x2, y2] format."""
    return torch.stack([
        boxes[..., 0] - boxes[..., 2] / 2,
        boxes[..., 1] - boxes[..., 3] / 2,
        boxes[..., 0] + boxes[..., 2] / 2,
        boxes[..., 1] + boxes[..., 3] / 2,
    ], dim=-1)


class DetrForDetection(GenericModel):
    """DETR model for object detection."""

    replace_keys_in_state_dict = {
        "conv_encoder\\.": "",
        "out_proj": "o_proj",
        r"(?<!mlp\.)fc1": "mlp.fc1",
        r"(?<!mlp\.)fc2": "mlp.fc2",
    }

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = DetrConfig.from_json_file(config_path)
        if num_classes is not None:
            self.config.num_labels = num_classes

        self.model = DetrModel(self.config)
        self.class_labels_classifier = nn.Linear(self.config.d_model, self.config.num_labels + 1)
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=self.config.d_model,
            hidden_dim=self.config.d_model,
            output_dim=4,
            num_layers=3,
        )
        self.init_weights()

        self.enable_auxiliary_loss = self.config.auxiliary_loss
        matcher = DetrHungarianMatcher(
            class_cost=self.config.class_cost,
            bbox_cost=self.config.bbox_cost,
            giou_cost=self.config.giou_cost,
        )
        self.criterion = DetrLoss(
            matcher=matcher,
            num_classes=self.config.num_labels,
            eos_coef=self.config.eos_coefficient,
            losses=["labels", "boxes", "cardinality"],
        )
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": self.config.bbox_loss_coefficient,
            "loss_giou": self.config.giou_loss_coefficient,
        }
        if self.enable_auxiliary_loss:
            aux_weight_dict = {
                f"{k}_{i}": v
                for i in range(self.config.decoder_layers - 1)
                for k, v in self.weight_dict.items()
            }
            self.weight_dict.update(aux_weight_dict)

    def _set_aux_loss(
        self, outputs_class: torch.Tensor, outputs_coord: torch.Tensor
    ) -> list:
        return [
            {"logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def forward(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            images = image_list_to_tensor(images)
        assert images.dim() == 4

        outputs = self.model(images.to(self.dtype))
        sequence_output = outputs[0]

        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        outputs_loss = {"logits": logits, "pred_boxes": pred_boxes}

        if self.enable_auxiliary_loss:
            intermediate = outputs[4]
            outputs_loss["auxiliary_outputs"] = self._set_aux_loss(
                self.class_labels_classifier(intermediate),
                self.bbox_predictor(intermediate).sigmoid(),
            )

        labels = [
            {"class_labels": c, "boxes": _xyxy_to_xywh(b)}
            for b, c in zip(bboxes, classes)
        ]
        loss_dict = self.criterion(outputs_loss, labels)
        return sum(
            loss_dict[k] * self.weight_dict[k]
            for k in loss_dict
            if k in self.weight_dict
        )

    @torch.no_grad()
    def detect(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        norm_bboxes: bool = False,
        threshold: float = 0.5,
    ) -> GenericOutputs:
        image_sizes = [(img.size(-2), img.size(-1)) for img in images]
        if not isinstance(images, torch.Tensor):
            images = image_list_to_tensor(images)
        assert images.dim() == 4

        outputs = self.model(images.to(self.dtype))
        logits = self.class_labels_classifier(outputs[0]).softmax(dim=-1)
        pred_boxes = [_xywh_to_xyxy(b) for b in self.bbox_predictor(outputs[0]).sigmoid()]

        scores, classes = zip(*[p.max(-1) for p in logits])

        if norm_bboxes:
            bboxes = pred_boxes
        else:
            bboxes = [
                b * torch.tensor([s[1], s[0], s[1], s[0]]).to(b)
                for b, s in zip(pred_boxes, image_sizes)
            ]

        keep = [
            (c != self.config.num_labels) & (s > threshold)
            for s, c in zip(scores, classes)
        ]
        bboxes = [b[m] for b, m in zip(bboxes, keep)]
        scores = [s[m] for s, m in zip(scores, keep)]
        classes = [c[m] for c, m in zip(classes, keep)]

        return GenericOutputs(bboxes=bboxes, scores=scores, classes=classes)
