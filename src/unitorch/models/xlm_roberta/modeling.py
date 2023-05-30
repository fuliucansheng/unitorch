# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaConfig,
    XLMRobertaModel,
)
from transformers.models.xlm_roberta_xl.modeling_xlm_roberta_xl import (
    XLMRobertaXLConfig,
    XLMRobertaXLModel,
)

from transformers.models.roberta.modeling_roberta import RobertaLMHead
from unitorch.models import GenericModel


class XLMRobertaForClassification(GenericModel):
    """
    XLM-RoBERTa model for classification tasks.
    """

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the XLMRobertaForClassification model.

        Args:
            config_path (str): Path to the configuration file.
            num_classes (Optional[int]): Number of classes. Defaults to 1.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = XLMRobertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = XLMRobertaModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the XLMRobertaForClassification model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [batch_size, sequence_length].
            attention_mask (Optional[torch.Tensor]): Attention mask tensor of shape [batch_size, sequence_length].
                Defaults to None.
            token_type_ids (Optional[torch.Tensor]): Token type IDs tensor of shape [batch_size, sequence_length].
                Defaults to None.
            position_ids (Optional[torch.Tensor]): Position IDs tensor of shape [batch_size, sequence_length].
                Defaults to None.

        Returns:
            (torch.Tensor):Output logits of shape [batch_size, num_classes].
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class XLMRobertaForMaskLM(GenericModel):
    """
    XLM-RoBERTa model for masked language modeling tasks.
    """

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the XLMRobertaForMaskLM model.

        Args:
            config_path (str): Path to the configuration file.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = XLMRobertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = XLMRobertaModel(self.config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(self.config)
        self.init_weights()
        self.roberta.embeddings.word_embeddings.weight = self.lm_head.decoder.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the XLMRobertaForMaskLM model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [batch_size, sequence_length].
            attention_mask (Optional[torch.Tensor]): Attention mask tensor of shape [batch_size, sequence_length].
                Defaults to None.
            token_type_ids (Optional[torch.Tensor]): Token type IDs tensor of shape [batch_size, sequence_length].
                Defaults to None.
            position_ids (Optional[torch.Tensor]): Position IDs tensor of shape [batch_size, sequence_length].
                Defaults to None.

        Returns:
            (torch.Tensor):Output logits of shape [batch_size, sequence_length, vocabulary_size].
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        return logits


class XLMRobertaXLForClassification(GenericModel):
    """
    XLM-RoBERTa XL model for classification tasks.
    """

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the XLMRobertaXLForClassification model.

        Args:
            config_path (str): Path to the configuration file.
            num_classes (Optional[int]): Number of classes for classification. Defaults to 1.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = XLMRobertaXLConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = XLMRobertaXLModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the XLMRobertaXLForClassification model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [batch_size, sequence_length].
            attention_mask (Optional[torch.Tensor]): Attention mask tensor of shape [batch_size, sequence_length].
                Defaults to None.
            token_type_ids (Optional[torch.Tensor]): Token type IDs tensor of shape [batch_size, sequence_length].
                Defaults to None.
            position_ids (Optional[torch.Tensor]): Position IDs tensor of shape [batch_size, sequence_length].
                Defaults to None.

        Returns:
            (torch.Tensor):Output logits of shape [batch_size, num_classes].
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
