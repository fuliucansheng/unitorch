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
from unitorch.utils import image_list_to_tensor


class GroundingDinoForDetection(GenericModel):
    replace_keys_in_state_dict = {}

    def __init__(
        self,
        config_path: str,
    ):
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
            for _ in range(self.config.decoder_layers):
                _bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=self.config.d_model,
                    hidden_dim=self.config.d_model,
                    output_dim=4,
                    num_layers=3,
                )
                self.bbox_embed = nn.ModuleList(
                    [_bbox_embed for _ in range(self.config.decoder_layers)]
                )
        self.class_embed = nn.ModuleList(
            [_class_embed for _ in range(self.config.decoder_layers)]
        )
        # hack for box-refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        # hack implementation for two-stage
        self.model.decoder.class_embed = self.class_embed
        self.init_weights()

        self.enable_auxiliary_loss = self.config.auxiliary_loss
        self.matcher = GroundingDinoHungarianMatcher(
            class_cost=self.config.class_cost,
            bbox_cost=self.config.bbox_cost,
            giou_cost=self.config.giou_cost,
        )
        # Second: create the criterion
        self.losses = ["labels", "boxes", "cardinality"]
        self.criterion = GroundingDinoLoss(
            matcher=self.matcher,
            num_classes=self.config.num_labels,
            focal_alpha=self.config.focal_alpha,
            losses=self.losses,
        )

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
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bboxes: Union[List[torch.Tensor], torch.Tensor],
        classes: Union[List[torch.Tensor], torch.Tensor],
    ):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text
        hidden_states = outputs.intermediate_hidden_states
        init_reference_points = outputs.init_reference_points
        inter_references_points = outputs.intermediate_reference_points
        outputs_classes = []
        outputs_coords = []
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])

            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(
                    f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}"
                )
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        # Third: compute the losses, based on outputs and labels
        outputs_loss = {}
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes

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

        if self.enable_auxiliary_loss:
            auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        if self.config.two_stage:
            enc_outputs_coord = outputs[-1].sigmoid()
            outputs_loss["enc_outputs"] = {
                "logits": outputs[-2],
                "pred_boxes": enc_outputs_coord,
            }

        loss_dict = self.criterion(outputs_loss, labels)
        # Fourth: compute total loss, as a weighted sum of the various losses
        weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
        weight_dict["loss_giou"] = self.config.giou_loss_coefficient
        if self.config.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(self.config.decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        return loss

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
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text
        hidden_states = outputs.intermediate_hidden_states
        init_reference_points = outputs.init_reference_points
        inter_references_points = outputs.intermediate_reference_points
        outputs_classes = []
        outputs_coords = []
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])

            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(
                    f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}"
                )
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        probs = torch.sigmoid(logits)
        scores = torch.max(probs, dim=-1)[0]
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

        outputs = dict(
            {
                "bboxes": list(bboxes),
                "scores": list(scores),
                "classes": list(classes),
            }
        )
        return GenericOutputs(outputs)
