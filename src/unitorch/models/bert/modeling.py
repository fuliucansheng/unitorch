# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from unitorch.models import GenericModel


class BertForClassification(GenericModel):
    """BERT model for sequence classification."""

    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}

    def __init__(
        self,
        config_path: str,
        num_classes: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
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
    ) -> torch.Tensor:
        pooled = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).pooler_output
        return self.classifier(self.dropout(pooled))
