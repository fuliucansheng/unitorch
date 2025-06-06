# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.megatron import (
    MegatronGPTForGeneration as _MegatronGPTForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs


@register_model("core/model/generation/megatron/gpt")
class MegatronGPTForGeneration(_MegatronGPTForGeneration):
    """Megatron model for generation tasks."""

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
        """
        Initialize the LlamaForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            normalization=normalization,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            num_experts=num_experts,
            use_transformer_engine=use_transformer_engine,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/megatron/gpt")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlamaForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlamaForGeneration: An instance of LlamaForGeneration.
        """
        config.set_default_section("core/model/generation/megatron/gpt")
        config_path = config.getoption("config_path", None)

        config_path = cached_path(config_path)
        vocab_size = config.getoption("vocab_size", 32000)
        num_experts = config.getoption("num_experts", None)

        inst = cls(
            config_path,
            vocab_size=vocab_size,
            num_experts=num_experts,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    def forward(
        self,
        input_ids: torch.Tensor = None,
        refs: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass on the LlamaForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input tensor containing the input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor. Defaults to None.

        Returns:
            GenerationOutputs: The output of the generation model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            input_ids_label=refs,
            attention_mask=attention_mask,
            attention_mask_label=masks,
        )
        if isinstance(outputs, torch.Tensor):
            return outputs
        return LossOutputs(loss=outputs.loss)
