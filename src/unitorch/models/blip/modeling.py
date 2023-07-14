# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import json
import math
import random
import logging
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.models.blip.modeling_blip import (
    BlipConfig,
    BlipTextModel,
    BlipVisionModel,
    BlipForConditionalGeneration,
)
from transformers.models.blip.modeling_blip_text import apply_chunking_to_forward
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.clip.modeling import AllGather


def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def _blip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = _contrastive_loss(similarity)
    image_loss = _contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


@replace(transformers.models.blip.modeling_blip_text.BlipTextSelfAttention)
class BlipTextSelfAttentionV2(
    transformers.models.blip.modeling_blip_text.BlipTextSelfAttention
):
    def __init__(
        self,
        config,
        is_cross_attention=False,
    ):
        super().__init__(
            config=config,
            is_cross_attention=is_cross_attention,
        )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        kv_bsz = bsz

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            kv_bsz = key_layer.size(0)
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if is_cross_attention and kv_bsz != bsz:
            attention_scores = torch.einsum(
                "bxhtd,bhsd->bxhts",
                query_layer.view(kv_bsz, -1, self.num_heads, *query_layer.size()[1:]),
                key_layer.view(kv_bsz, self.num_heads, *key_layer.size()[1:]),
            )
            attention_scores = attention_scores.reshape(
                -1, *attention_scores.size()[-2:]
            )
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                if kv_bsz != bsz:
                    shape = relative_position_scores_query.size()
                    relative_position_scores_key = (
                        relative_position_scores_key.unsqueeze(1)
                        .expand(kv_bsz, bsz // kv_bsz, *shape[1:])
                        .contiguous()
                        .view(bsz, *shape[1:])
                    )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BlipTextModel forward() function)
            attention_scores = attention_scores + attention_mask.to(
                attention_scores.device
            )

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        if is_cross_attention and kv_bsz != bsz:
            context_layer = torch.einsum(
                "bxhtd,bhsd->bxhts",
                attention_probs_dropped.view(
                    kv_bsz, -1, self.num_heads, *query_layer.size()[1:]
                ),
                value_layer.view(kv_bsz, self.num_heads, *key_layer.size()[1:]),
            )
            context_layer = context_layer.reshape(-1, *context_layer.size()[-2:])
        else:
            context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        outputs = outputs + (past_key_value,)
        return outputs


@replace(transformers.models.blip.modeling_blip_text.BlipTextLayer)
class BlipTextLayerV2(transformers.models.blip.modeling_blip_text.BlipTextLayer):
    def __init__(self, config, layer_num):
        super().__init__(
            config=config,
            layer_num=layer_num,
        )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if encoder_hidden_states is not None:
            cross_attn_past_key_value = (
                past_key_value[2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                past_key_value=cross_attn_past_key_value,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights
            present_key_value = present_key_value + cross_attention_outputs[-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs


class BlipForPretrain(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        """
        Initializes the BlipForPretrain model.

        Args:
            config_path (str): Path to the configuration file.
            projection_dim (Optional[int], optional): Dimension of the projection. Defaults to 512.
            freeze_base_model (Optional[bool], optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (Optional[bool], optional): Whether to use gradient checkpointing. Defaults to False.
            use_all_gather (Optional[bool], optional): Whether to use all_gather operation for distributed training. Defaults to True.
        """
        super().__init__()

        config = BlipConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.use_all_gather = use_all_gather

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = BlipTextModel(text_config)
        self.vision_model = BlipVisionModel(vision_config)

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
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def _all_gather(self, input):
        """
        Applies all_gather operation for distributed training.

        Args:
            input: Input tensor to gather.

        Returns:
            output: Gathered tensor across all workers.
        """
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the BlipForPretrain model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Pixel values of the images.
            attention_mask (torch.Tensor): Attention mask for the input.
            position_ids (torch.Tensor): Position IDs for the input.

        Returns:
            (torch.Tensor):Output loss for the pretraining task.
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

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self._all_gather(text_embeds)
            image_embeds = self._all_gather(image_embeds)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        return _blip_loss(logits_per_text)


class BlipForClassification(nn.Module):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the BlipForClassification model.

        Args:
            config_path (str): Path to the configuration file.
            projection_dim (Optional[int], optional): Dimension of the projection. Defaults to 512.
            num_classes (Optional[int], optional): Number of classes for classification. Defaults to 1.
            freeze_base_model (Optional[bool], optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (Optional[bool], optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # Load the BLIP model configuration
        config = BlipConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config

        # Set gradient checkpointing option
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # Initialize the text and vision models
        self.text_model = BlipTextModel(text_config)
        self.vision_model = BlipVisionModel(vision_config)

        # Projection layers for text and vision embeddings
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

        # Classifier layer
        self.classifier = nn.Linear(self.projection_dim * 2, num_classes)

        # Initialize the weights of the model
        self.init_weights()

        if freeze_base_model:
            # Freeze the parameters of the base models if specified
            for p in self.text_model.parameters():
                p.requires_grad = False

            for p in self.vision_model.parameters():
                p.requires_grad = False

        # Set gradient checkpointing option for encoders
        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing
        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the BlipForClassification model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Pixel values of the images.
            attention_mask (torch.Tensor): Attention mask for the input.
            position_ids (torch.Tensor): Position IDs for the input.

        Returns:
            (torch.Tensor):Output logits for classification.
        """
        # Process the vision modality
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        # Process the text modality
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Project vision embeddings to the specified dimensionality
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # Project text embeddings to the specified dimensionality
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # Concatenate and classify the projected embeddings
        return self.classifier(F.relu(torch.cat([image_embeds, text_embeds], axis=1)))


class BlipForTextClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the BlipForTextClassification model.

        Args:
            config_path (str): Path to the configuration file.
            projection_dim (Optional[int], optional): Dimension of the projection. Defaults to 512.
            num_classes (Optional[int], optional): Number of classes for classification. Defaults to 1.
            freeze_base_model (Optional[bool], optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (Optional[bool], optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # Load the BLIP model configuration
        config = BlipConfig.from_json_file(config_path)
        text_config = config.text_config
        text_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size

        # Initialize the BLIP text model
        self.text_model = BlipTextModel(text_config)

        # Project text embeddings to the desired dimension
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )

        # Classifier layer for classification task
        self.classifier = nn.Linear(self.projection_dim, num_classes)

        # Initialize the model weights
        self.init_weights()

        # Freeze the base model if specified
        if freeze_base_model:
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Set gradient checkpointing for the encoder
        self.text_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Forward pass of the BlipForTextClassification model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            position_ids (torch.Tensor): Position IDs for the input.

        Returns:
            (torch.Tensor):Output logits for classification.
        """
        # Pass the input through the BLIP text model
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Extract text embeddings and project to desired dimension
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # Apply ReLU activation and pass through the classifier layer
        return self.classifier(nn.functional.relu(text_embeds))


class BlipForImageClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the BlipForImageClassification model.

        Args:
            config_path (str): Path to the configuration file.
            projection_dim (Optional[int], optional): Dimension of the projection. Defaults to 512.
            num_classes (Optional[int], optional): Number of classes for classification. Defaults to 1.
            freeze_base_model (Optional[bool], optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (Optional[bool], optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # Load the BLIP model configuration
        config = BlipConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.projection_dim = projection_dim
        self.vision_embed_dim = vision_config.hidden_size

        self.vision_model = BlipVisionModel(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(self.projection_dim, num_classes)

        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Forward pass of the BlipForImageClassification model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            (torch.Tensor):Output logits for classification.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[1]

        image_embeds = self.visual_projection(image_embeds)

        image_embeds = F.relu(image_embeds)

        return self.classifier(image_embeds)


class BlipForImageCaption(GenericModel):
    prefix_keys_in_state_dict = {"^(?!model\.).*": "model."}

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes the BlipForImageCaption model.

        Args:
            config_path (str): Path to the configuration file.
            gradient_checkpointing (Optional[bool], optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = BlipConfig.from_json_file(config_path)
        self.config.vision_config.gradient_checkpointing = gradient_checkpointing
        self.config.text_config.gradient_checkpointing = gradient_checkpointing
        self.model = BlipForConditionalGeneration(self.config)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the BlipForImageCaption model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.
            input_ids (Optional[torch.Tensor], optional): Input token IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.

        Returns:
            (torch.Tensor):Logits for caption generation.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.decoder_logits
        return logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 101,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 102,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 48,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        """
        Generates captions for the given input images.

        Args:
            pixel_values (torch.Tensor): Input pixel values.
            input_ids (Optional[torch.Tensor], optional): Input token IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            num_beams (Optional[int], optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (Optional[int], optional): ID of the start token for decoding. Defaults to 30522.
            decoder_end_token_id (int or List[int], optional): ID of the end token for decoding. Defaults to 2.
            num_return_sequences (Optional[int], optional): Number of caption sequences to return. Defaults to 1.
            min_gen_seq_length (Optional[int], optional): Minimum length of generated sequences. Defaults to 0.
            max_gen_seq_length (Optional[int], optional): Maximum length of generated sequences. Defaults to 48.
            repetition_penalty (Optional[float], optional): Repetition penalty value. Defaults to 1.0.
            no_repeat_ngram_size (Optional[int], optional): Size of n-grams to avoid repetition. Defaults to 0.
            early_stopping (Optional[bool], optional): Whether to stop generation early. Defaults to True.
            length_penalty (Optional[float], optional): Length penalty value. Defaults to 1.0.
            num_beam_groups (Optional[int], optional): Number of groups for diverse beam search. Defaults to 1.
            diversity_penalty (Optional[float], optional): Diversity penalty value. Defaults to 0.0.
            do_sample (Optional[bool], optional): Whether to use sampling for generation. Defaults to False.
            temperature (Optional[float], optional): Temperature value for sampling. Defaults to 1.0.
            top_k (Optional[int], optional): Value of k for top-k sampling. Defaults to 50.
            top_p (Optional[float], optional): Value of p for top-p sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: Generated caption sequences and their scores.
        """
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_seq_length,
            min_length=min_gen_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            decoder_start_token_id=decoder_start_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = outputs.sequences.reshape(
            -1, num_return_sequences, outputs.sequences.size(-1)
        )
        outputs.sequences = torch.zeros(
            sequences.size(0), num_return_sequences, max_gen_seq_length
        ).to(device=sequences.device)
        outputs.sequences[:, :, : sequences.size(-1)].copy_(sequences)

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=outputs.sequences, sequences_scores=outputs.sequences_scores
        )
