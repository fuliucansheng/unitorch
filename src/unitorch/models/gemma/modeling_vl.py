# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import copy
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.gemma4 import Gemma4Config, Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.gemma.modeling import _get_gemma_dtype


def _reshape_image_inputs(
    pixel_values: Optional[torch.Tensor],
    image_position_ids: Optional[torch.Tensor],
):
    if pixel_values is not None and pixel_values.dim() == 4:
        pixel_values = pixel_values.view(
            -1,
            pixel_values.size(-2),
            pixel_values.size(-1),
        )
    if image_position_ids is not None and image_position_ids.dim() == 4:
        image_position_ids = image_position_ids.view(
            -1,
            image_position_ids.size(-2),
            image_position_ids.size(-1),
        )
    return pixel_values, image_position_ids


class GemmaUnifiedVisionTower(nn.Module):
    """
    Encoder-free vision tower used by Gemma 4 12B unified checkpoints.
    """

    def __init__(
        self,
        patch_size: int,
        pooling_kernel_size: int,
        hidden_size: int,
        max_soft_tokens: int = 1120,
    ):
        super().__init__()
        patch_dim = 3 * patch_size**2
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_ln1 = nn.LayerNorm(patch_dim * pooling_kernel_size**2)
        self.patch_dense = nn.Linear(
            patch_dim * pooling_kernel_size**2,
            hidden_size,
        )
        self.patch_ln2 = nn.LayerNorm(hidden_size)
        self.pos_embedding = nn.Parameter(
            torch.zeros(max_soft_tokens, 2, hidden_size)
        )
        self.pos_norm = nn.LayerNorm(hidden_size)

    def _pool_image_patches(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ):
        valid_mask = ~(pixel_position_ids == -1).all(dim=-1)
        if not valid_mask.any():
            return (
                pixel_values.new_empty((0, self.patch_dense.in_features)),
                pixel_position_ids.new_empty((0, 2)),
            )

        patches = pixel_values[valid_mask]
        positions = pixel_position_ids[valid_mask]
        pooled_positions = torch.div(
            positions,
            self.pooling_kernel_size,
            rounding_mode="floor",
        )
        num_pooled_x = int(pooled_positions[:, 0].max().item()) + 1
        cell_ids = pooled_positions[:, 1] * num_pooled_x + pooled_positions[:, 0]
        num_cells = int(cell_ids.max().item()) + 1
        expected_num_cells = patches.size(0) // (self.pooling_kernel_size**2)
        if expected_num_cells != num_cells:
            raise ValueError(
                "Gemma unified vision pooling expected contiguous 3x3 patch groups "
                f"but received {patches.size(0)} patches mapped to {num_cells} cells."
            )

        local_offsets = torch.remainder(positions, self.pooling_kernel_size)
        local_ids = local_offsets[:, 1] * self.pooling_kernel_size + local_offsets[:, 0]
        grouped = patches.new_zeros(
            (num_cells, self.pooling_kernel_size**2, patches.size(-1))
        )
        grouped[cell_ids.long(), local_ids.long()] = patches

        pooled_token_positions = torch.stack(
            [
                torch.arange(num_cells, device=positions.device) % num_pooled_x,
                torch.div(
                    torch.arange(num_cells, device=positions.device),
                    num_pooled_x,
                    rounding_mode="floor",
                ),
            ],
            dim=-1,
        )
        return grouped.reshape(num_cells, -1), pooled_token_positions

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        pooled_patches, pooled_positions = [], []
        for image_patches, image_positions in zip(pixel_values, pixel_position_ids):
            patches, positions = self._pool_image_patches(image_patches, image_positions)
            if patches.numel() == 0:
                continue
            pooled_patches.append(patches)
            pooled_positions.append(positions)

        if len(pooled_patches) == 0:
            hidden_states = pixel_values.new_empty((0, self.patch_dense.out_features))
            return BaseModelOutputWithPooling(last_hidden_state=hidden_states)

        pooled_patches = torch.cat(pooled_patches, dim=0)
        pooled_positions = torch.cat(pooled_positions, dim=0)
        pooled_patches = pooled_patches.to(
            device=self.patch_ln1.weight.device,
            dtype=self.patch_ln1.weight.dtype,
        )
        pooled_positions = pooled_positions.to(device=self.pos_embedding.device)

        hidden_states = self.patch_ln1(pooled_patches)
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        pooled_positions = pooled_positions.clamp_(0, self.pos_embedding.size(0) - 1)
        position_embeddings = (
            self.pos_embedding[pooled_positions[:, 0], 0]
            + self.pos_embedding[pooled_positions[:, 1], 1]
        )
        position_embeddings = position_embeddings.to(
            dtype=self.pos_norm.weight.dtype
        )
        hidden_states = hidden_states + self.pos_norm(position_embeddings).to(
            hidden_states.dtype
        )
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class GemmaVLForGeneration(GenericModel, PeftWeightLoaderMixin):
    """
    Gemma vision-language generation model backed by Gemma4's unified checkpoint.
    """

    prefix_keys_in_state_dict = {"^(?!model\\.model\\.).*": "model."}
    replace_keys_in_state_dict = {
        r"model\.model\.vision_embedder\.": "model.model.vision_tower.",
    }

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = Gemma4Config.from_json_file(config_path)
        self.vision_config = copy.deepcopy(self.config.vision_config)
        self.config.audio_config = None
        self.config.vision_config = None
        if gradient_checkpointing:
            self.config.use_cache = False
            if getattr(self.config, "text_config", None) is not None:
                self.config.text_config.use_cache = False
        self.model = Gemma4ForConditionalGeneration(self.config)
        self.model.model.vision_tower = GemmaUnifiedVisionTower(
            patch_size=self.vision_config.patch_size,
            pooling_kernel_size=self.vision_config.pooling_kernel_size,
            hidden_size=self.vision_config.output_proj_dims,
        )
        self.model.model.embed_vision = Gemma4MultimodalEmbedder(
            self.vision_config,
            self.config.text_config,
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.init_weights()
        self.model.to(dtype=_get_gemma_dtype(self.config))

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ):
        pixel_values, image_position_ids = _reshape_image_inputs(
            pixel_values,
            image_position_ids,
        )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            return_dict=True,
        )
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 2,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 1,
        decoder_pad_token_id: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 512,
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
        input_seq_length = input_ids.size(1)
        pixel_values, image_position_ids = _reshape_image_inputs(
            pixel_values,
            image_position_ids,
        )
        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            max_length=max_gen_seq_length + input_seq_length,
            min_length=min_gen_seq_length + input_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
            pad_token_id=decoder_pad_token_id,
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
        padded = torch.full(
            (sequences.size(0), num_return_sequences, max_gen_seq_length),
            fill_value=decoder_pad_token_id,
            device=sequences.device,
        )
        padded[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            padded = padded.reshape(-1, max_gen_seq_length)

        return GenericOutputs(
            sequences=padded.long(),
            sequences_scores=getattr(outputs, "sequences_scores", None),
        )
