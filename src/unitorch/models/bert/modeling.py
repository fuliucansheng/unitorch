# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import BertConfig, BertModel
from unitorch.models import GenericModel


class BertForClassification(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the BertForClassification model.

        Args:
            config_path (str): The path to the configuration file.
            num_classes (Optional[int], optional): The number of classes for classification. Defaults to 1.
            gradient_checkpointing (Optional[bool], optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.bert = BertModel(self.config)
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
        Forward pass of the BertForClassification model.

        Args:
            input_ids (torch.Tensor): The input tensor of token indices.
            attention_mask (torch.Tensor optional): The attention mask tensor. Defaults to None.
            token_type_ids (torch.Tensor optional): The token type IDs tensor. Defaults to None.
            position_ids (torch.Tensor optional): The position IDs tensor. Defaults to None.

        Returns:
            (torch.Tensor):The logits of the model output.
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
