# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from transformers.models.grounding_dino import GroundingDinoConfig
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoModel,
    GroundingDinoContrastiveEmbedding,
    GroundingDinoMLPPredictionHead,
)
from transformers.loss.loss_deformable_detr import (
    DeformableDetrHungarianMatcher as GroundingDinoHungarianMatcher,
    DeformableDetrImageLoss as GroundingDinoLoss,
)
from unitorch.models import GenericModel, GenericOutputs


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format."""
    return torch.stack(
        [
            (boxes[..., 0] + boxes[..., 2]) / 2,
            (boxes[..., 1] + boxes[..., 3]) / 2,
            boxes[..., 2] - boxes[..., 0],
            boxes[..., 3] - boxes[..., 1],
        ],
        dim=-1,
    )


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    return torch.stack(
        [
            boxes[..., 0] - boxes[..., 2] / 2,
            boxes[..., 1] - boxes[..., 3] / 2,
            boxes[..., 0] + boxes[..., 2] / 2,
            boxes[..., 1] + boxes[..., 3] / 2,
        ],
        dim=-1,
    )


class GroundingDinoForDetection(GenericModel):
    replace_keys_in_state_dict = {}

    def __init__(
        self,
        config_path: str,
    ):
        """
        Initializes the GroundingDinoForDetection model.

        Args:
            config_path (str): Path to the GroundingDINO configuration file.
        """
        super().__init__()
        self.config = GroundingDinoConfig.from_json_file(config_path)
        self.model = GroundingDinoModel(self.config)
        _class_embed = GroundingDinoContrastiveEmbedding(self.config)

        if self.config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=self.config.d_model,
                hidden_dim=self.config.d_model,
                output_dim=4,
                num_layers=3,
            )
            self.bbox_embed = nn.ModuleList(
                [_bbox_embed for _ in range(self.config.decoder_layers)]
            )
        else:
            self.bbox_embed = nn.ModuleList(
                [
                    GroundingDinoMLPPredictionHead(
                        input_dim=self.config.d_model,
                        hidden_dim=self.config.d_model,
                        output_dim=4,
                        num_layers=3,
                    )
                    for _ in range(self.config.decoder_layers)
                ]
            )

        self.class_embed = nn.ModuleList(
            [_class_embed for _ in range(self.config.decoder_layers)]
        )
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        self.init_weights()

        self.enable_auxiliary_loss = self.config.auxiliary_loss
        self.matcher = GroundingDinoHungarianMatcher(
            class_cost=self.config.class_cost,
            bbox_cost=self.config.bbox_cost,
            giou_cost=self.config.giou_cost,
        )
        self.losses = ["labels", "boxes", "cardinality"]
        self.criterion = GroundingDinoLoss(
            matcher=self.matcher,
            num_classes=self.config.num_labels,
            focal_alpha=self.config.focal_alpha,
            losses=self.losses,
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def _set_aux_loss(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
    ) -> List[Dict]:
        return [
            {"logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def _decode_outputs(
        self,
        hidden_states: torch.Tensor,
        enc_text_hidden_state: torch.Tensor,
        init_reference_points: torch.Tensor,
        inter_references_points: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Decodes model outputs into class and coordinate predictions."""
        outputs_classes = []
        outputs_coords = []
        num_levels = hidden_states.shape[1]

        for level in range(num_levels):
            reference = init_reference_points if level == 0 else inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)

            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])

            ref_dim = reference.shape[-1]
            if ref_dim == 4:
                outputs_coord_logits = delta_bbox + reference
            elif ref_dim == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(
                    f"reference.shape[-1] should be 4 or 2, but got {ref_dim}"
                )

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord_logits.sigmoid())

        return torch.stack(outputs_classes), torch.stack(outputs_coords)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ):
        """
        Forward pass computing detection loss.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            input_ids (torch.Tensor): Text token IDs.
            token_type_ids (torch.Tensor): Token type IDs.
            attention_mask (torch.Tensor): Attention mask.
            bboxes (List[torch.Tensor] or torch.Tensor): Ground-truth boxes in xyxy format.
            classes (List[torch.Tensor] or torch.Tensor): Ground-truth class IDs.

        Returns:
            torch.Tensor: Total detection loss.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        outputs_class, outputs_coord = self._decode_outputs(
            hidden_states=outputs.intermediate_hidden_states,
            enc_text_hidden_state=outputs.encoder_last_hidden_state_text,
            init_reference_points=outputs.init_reference_points,
            inter_references_points=outputs.intermediate_reference_points,
            attention_mask=attention_mask,
        )

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        bboxes = [xyxy2xywh(bbox) for bbox in bboxes]
        labels = [{"class_labels": c, "boxes": b} for b, c in zip(bboxes, classes)]

        outputs_loss = {"logits": logits, "pred_boxes": pred_boxes}
        if self.enable_auxiliary_loss:
            outputs_loss["auxiliary_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.config.two_stage:
            outputs_loss["enc_outputs"] = {
                "logits": outputs[-2],
                "pred_boxes": outputs[-1].sigmoid(),
            }

        loss_dict = self.criterion(outputs_loss, labels)
        weight_dict = {
            "loss_ce": 1,
            "loss_bbox": self.config.bbox_loss_coefficient,
            "loss_giou": self.config.giou_loss_coefficient,
        }
        if self.config.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(self.config.decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        return sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )

    @torch.no_grad()
    def detect(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        norm_bboxes: Optional[bool] = False,
        text_threshold: Optional[float] = 0.25,
        box_threshold: Optional[float] = 0.25,
    ):
        """
        Runs detection inference and returns filtered predictions.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            input_ids (torch.Tensor): Text token IDs.
            token_type_ids (torch.Tensor): Token type IDs.
            attention_mask (torch.Tensor): Attention mask.
            norm_bboxes (bool, optional): Whether to return normalized boxes. Defaults to False.
            text_threshold (float, optional): Threshold for text token scores. Defaults to 0.25.
            box_threshold (float, optional): Threshold for box confidence scores. Defaults to 0.25.

        Returns:
            GenericOutputs: Detected bounding boxes, scores, and class IDs.
        """
        h, w = pixel_values.shape[-2:]
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        outputs_class, outputs_coord = self._decode_outputs(
            hidden_states=outputs.intermediate_hidden_states,
            enc_text_hidden_state=outputs.encoder_last_hidden_state_text,
            init_reference_points=outputs.init_reference_points,
            inter_references_points=outputs.intermediate_reference_points,
            attention_mask=attention_mask,
        )

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        probs = torch.sigmoid(logits)
        scores = torch.max(probs, dim=-1)[0]
        pred_boxes = [xywh2xyxy(bbox) for bbox in pred_boxes]

        if not norm_bboxes:
            bboxes = [b * torch.tensor([w, h, w, h]).to(b) for b in pred_boxes]
        else:
            bboxes = pred_boxes

        class_ids = input_ids.unsqueeze(1).expand(-1, bboxes[0].shape[0], -1)
        max_len = min(input_ids.shape[-1], logits.shape[-1])

        bboxes, scores, classes = list(
            zip(
                *[
                    (
                        b[s > box_threshold],
                        s[s > box_threshold],
                        (
                            (p[s > box_threshold] > text_threshold).float()[:, :max_len]
                            * c[s > box_threshold].float()[:, :max_len]
                        ).long(),
                    )
                    for b, s, p, c in zip(bboxes, scores, probs, class_ids)
                ]
            )
        )

        return GenericOutputs(
            bboxes=list(bboxes),
            scores=list(scores),
            classes=list(classes),
        )
