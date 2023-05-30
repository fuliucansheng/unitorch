# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.visual_bert.modeling_visual_bert import (
    VisualBertConfig,
    VisualBertModel,
    VisualBertPreTrainingHeads,
)
from unitorch.models import GenericModel


class VisualBertForClassification(GenericModel):
    """
    VisualBERT model for classification tasks.
    """

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the VisualBertForClassification model.

        Args:
            config_path (str): The path to the VisualBERT model config file.
            num_classes (int, optional): The number of output classes for classification. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = VisualBertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.visual_bert = VisualBertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        visual_token_type_ids: torch.Tensor,
    ):
        """
        Forward pass of the VisualBertForClassification model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.
            position_ids (torch.Tensor): The position IDs.
            visual_embeds (torch.Tensor): The visual embeddings.
            visual_attention_mask (torch.Tensor): The visual attention mask.
            visual_token_type_ids (torch.Tensor): The visual token type IDs.

        Returns:
            (torch.Tensor): The logits for classification.
        """
        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class VisualBertForPretrain(GenericModel):
    """
    VisualBERT model for pretraining tasks.
    """

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the VisualBertForPretrain model.

        Args:
            config_path (str): The path to the VisualBERT model config file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = VisualBertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.visual_bert = VisualBertModel(self.config)
        self.cls = VisualBertPreTrainingHeads(self.config)
        self.init_weights()

        self.mlm_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.nsp_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def get_output_embeddings(self):
        """
        Get the output embeddings of the model.

        Returns:
            (nn.Module): The output embeddings.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings of the model.

        Args:
            new_embeddings (nn.Module): The new output embeddings.
        """
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        visual_token_type_ids: torch.Tensor,
        nsp_label: torch.Tensor,
        mlm_label: torch.Tensor,
        mlm_label_mask: torch.Tensor,
    ):
        """
        Forward pass of the VisualBertForPretrain model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.
            position_ids (torch.Tensor): The position IDs.
            visual_embeds (torch.Tensor): The visual embeddings.
            visual_attention_mask (torch.Tensor): The visual attention mask.
            visual_token_type_ids (torch.Tensor): The visual token type IDs.
            nsp_label (torch.Tensor): The next sentence prediction labels.
            mlm_label (torch.Tensor): The masked language modeling labels.
            mlm_label_mask (torch.Tensor): The masked language modeling label mask.

        Returns:
            (torch.Tensor): The loss of the model.
        """
        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        batch_size, seq_len, vocab_size = prediction_scores.size()
        masked_lm_loss = self.mlm_loss_fn(
            prediction_scores.view(-1, vocab_size), mlm_label.view(-1)
        ) * mlm_label_mask.view(-1)
        masked_lm_loss = masked_lm_loss.view(batch_size, seq_len).sum(1) / torch.max(
            mlm_label_mask.view(batch_size, seq_len).sum(1),
            torch.ones(batch_size).to(mlm_label_mask.device),
        )
        loss = masked_lm_loss.mean()

        loss += self.nsp_loss_fn(
            seq_relationship_score.view(-1, 2), nsp_label.view(-1)
        ).mean()

        return loss
