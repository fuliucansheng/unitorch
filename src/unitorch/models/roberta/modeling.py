# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
    RobertaLMHead,
)
from unitorch.models import GenericModel


class RobertaForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the RobertaForClassification model.

        Args:
            config_path (str): Path to the Roberta configuration file.
            num_classes (int, optional): Number of output classes. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = RobertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = RobertaModel(self.config)
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
        Forward pass of the RobertaForClassification model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            token_type_ids (torch.Tensor, optional): Token type IDs. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            torch.Tensor: Classification logits.
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = self.dropout(outputs[1])
        return self.classifier(pooled_output)


class RobertaForMaskLM(GenericModel):
    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the RobertaForMaskLM model.

        Args:
            config_path (str): Path to the Roberta configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = RobertaConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.roberta = RobertaModel(self.config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(self.config)
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the RobertaForMaskLM model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            token_type_ids (torch.Tensor, optional): Token type IDs. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            torch.Tensor: MLM logits of shape [batch_size, seq_len, vocab_size].
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return self.lm_head(outputs[0])
