# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import logging
import json
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import transformers
import torch.distributed as dist
from torch import autocast
from transformers.utils import is_remote_url
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.enums import ModelType
from unitorch import hf_cached_path
from unitorch.utils import read_json_file
from unitorch.utils.decorators import replace
from unitorch.models import GenericModel, GenericOutputs
from unitorch.models.megatron import GenericMegatronModel


class MegatronGPTForGeneration(GenericMegatronModel):
    """
    Megatron GPT Model
    """

    modules_to_save_checkpoints = ["model"]

    def __init__(
        self,
        config_path: str,
        vocab_size: int,
        max_position_embeddings: Optional[int] = 512,
        normalization: Optional[str] = "RMSNorm",
        position_embedding_type: Optional[str] = "rope",
        rotary_percent: Optional[float] = 0.5,
        rotary_base: Optional[int] = 10000,
        rope_scaling: Optional[bool] = False,
        num_experts: Optional[int] = None,
        use_transformer_engine: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TransformerConfig(
            **read_json_file(config_path),
            tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
            pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
            virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
            context_parallel_size=mpu.get_context_parallel_world_size(),
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            autocast_dtype=torch.bfloat16,
        )
        self.use_transformer_engine = use_transformer_engine
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.is_pp_first_rank = mpu.is_pipeline_first_stage(ignore_virtual=True)
        self.is_pp_last_rank = mpu.is_pipeline_last_stage(ignore_virtual=True)
        use_preprocess, use_postprocess = self.is_pp_first_rank, self.is_pp_last_rank

        if num_experts is not None:
            if self.use_transformer_engine:
                self.layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    num_experts=num_experts,
                    moe_grouped_gemm=True,
                    qk_layernorm=True,
                )
            else:
                self.layer_spec = get_gpt_layer_local_spec(
                    num_experts=num_experts,
                    moe_grouped_gemm=True,
                    qk_layernorm=True,
                    normalization=normalization,
                )
        self.model = GPTModel(
            config=self.config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                self.config,
                use_transformer_engine=use_transformer_engine,
                normalization=normalization,
            ),
            pre_process=use_preprocess,
            post_process=use_postprocess,
            vocab_size=vocab_size,
            max_sequence_length=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            mtp_block_spec=get_gpt_mtp_block_spec(
                self.config,
                self.layer_spec,
                use_transformer_engine=use_transformer_engine,
            )
            if num_experts is not None
            else None,
        )
        self.model_type = self.model.model_type
        self.model = self.model.to(torch.bfloat16)

    def set_input_tensor(self, input_tensor):
        self.model.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: torch.Tensor = None,
        input_ids_label: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_label: Optional[torch.Tensor] = None,
    ):
        if self.is_pp_first_rank:
            batch_size, seq_len = input_ids.size()
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
            causal_mask = torch.tril(
                torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=attention_mask.device
                )
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                0
            )  # shape (1, 1, seq_len, seq_len)
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            pad_mask = pad_mask.expand(-1, -1, seq_len, -1)
            attn_mask_4d = (
                causal_mask & pad_mask
            )  # shape: (batch_size, 1, seq_len, seq_len)
            attn_mask_4d = attn_mask_4d.expand(
                batch_size,
                self.config.num_attention_heads // self.tp_size,
                seq_len,
                seq_len,
            )
            output_tensor = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask_4d.bool(),
                position_ids=position_ids,
                labels=input_ids_label if self.is_pp_last_rank else None,
            )
        else:
            output_tensor = self.model(
                decoder_input=None,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                labels=input_ids_label,
            )

        if self.is_pp_last_rank:
            losses = output_tensor.float()
            loss_mask = attention_mask_label.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            return GenericOutputs(loss=loss)

        return output_tensor
