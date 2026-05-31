# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import List, Optional, Union
from peft import LoraConfig
from transformers.models.clip.modeling_clip import (
    CLIPConfig,
    CLIPTextModel,
    CLIPVisionModel,
)
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import (
    PeftModelForSequenceClassification,
    GenericPeftModel,
    PeftWeightLoaderMixin,
)


class ClipForMatching(GenericModel):
    """CLIP model for image-text matching."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
    ):
        """
        Initializes the ClipForMatching model.

        Args:
            config_path (str): Path to the model configuration file.
            projection_dim (int, optional): Dimension of the projected embeddings. Defaults to 512.
        """
        super().__init__()

        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextModel(text_config)
        self.vision_model = CLIPVisionModel(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the ClipForMatching model.

        Args:
            input_ids (torch.Tensor): Input text token IDs.
            pixel_values (torch.Tensor): Input image pixel values.
            attention_mask (torch.Tensor): Attention mask for text input.
            position_ids (torch.Tensor): Position IDs for text input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (text_embeds, image_embeds).
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        return (text_embeds, image_embeds)


class ClipForTextMatching(GenericModel):
    """CLIP model for text-to-text matching."""

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
    ):
        """
        Initializes the ClipForTextMatching model.

        Args:
            config_path (str): Path to the model configuration file.
            projection_dim (int, optional): Dimension of the projected embeddings. Defaults to 512.
        """
        super().__init__()

        config = CLIPConfig.from_json_file(config_path)
        text_config = config.text_config

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size

        self.query_model = CLIPTextModel(text_config)
        self.doc_model = CLIPTextModel(text_config)

        self.query_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.doc_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.init_weights()

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_position_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        doc_position_ids: torch.Tensor,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the ClipForTextMatching model.

        Args:
            query_input_ids (torch.Tensor): Query text token IDs.
            query_attention_mask (torch.Tensor): Query attention mask.
            query_position_ids (torch.Tensor): Query position IDs.
            doc_input_ids (torch.Tensor): Document text token IDs.
            doc_attention_mask (torch.Tensor): Document attention mask.
            doc_position_ids (torch.Tensor): Document position IDs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (query_embeds, doc_embeds).
        """

        query_outputs = self.query_model(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            position_ids=query_position_ids,
        )

        query_embeds = query_outputs[1]
        query_embeds = self.query_projection(query_embeds)

        doc_outputs = self.doc_model(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            position_ids=doc_position_ids,
        )

        doc_embeds = doc_outputs[1]
        doc_embeds = self.doc_projection(doc_embeds)

        return (query_embeds, doc_embeds)


class ClipLoraForMatching(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "q_proj.weight": "q_proj.base_layer.weight",
        "q_proj.bias": "q_proj.base_layer.bias",
        "v_proj.weight": "v_proj.base_layer.weight",
        "v_proj.bias": "v_proj.base_layer.bias",
    }
    modules_to_save_checkpoints = ["lora", "classifier"]
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            ClipForMatching(config_path, projection_dim=projection_dim),
            self.peft_config,
        )
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        text_embeds, image_embeds = self.peft_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return outputs


class ClipLoraForTextMatching(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "q_proj.weight": "q_proj.base_layer.weight",
        "q_proj.bias": "q_proj.base_layer.bias",
        "v_proj.weight": "v_proj.base_layer.weight",
        "v_proj.bias": "v_proj.base_layer.bias",
    }
    modules_to_save_checkpoints = ["lora", "classifier"]
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }

    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            ClipForTextMatching(config_path, projection_dim=projection_dim),
            self.peft_config,
        )
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_position_ids: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        doc_position_ids: torch.Tensor,
    ):
        query_embeds, doc_embeds = self.peft_model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            query_position_ids=query_position_ids,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            doc_position_ids=doc_position_ids,
        )

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(query_embeds * doc_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return outputs
