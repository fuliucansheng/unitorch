# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.peft import (
    LlamaAdaLoraForClassification as _LlamaAdaLoraForClassification,
    LlamaAdaLoraForGeneration as _LlamaAdaLoraForGeneration,
    LlamaLoraForClassification as _LlamaLoraForClassification,
    LlamaLoraForGeneration as _LlamaLoraForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.llama import pretrained_llama_infos


@register_model("core/model/classification/peft/adalora/llama")
class LlamaAdaLoraForClassification(_LlamaAdaLoraForClassification):
    """LlamaAdaLora model for classification tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        target_r: Optional[int] = 8,
        init_r: Optional[int] = 12,
        tinit: Optional[int] = 0,
        tfinal: Optional[int] = 0,
        deltaT: Optional[int] = 1,
        beta1: Optional[float] = 0.85,
        beta2: Optional[float] = 0.85,
        orth_reg_weight: Optional[float] = 0.5,
        total_step: Optional[int] = None,
        rank_pattern: Optional[dict] = None,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlamaAdaLoraForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            target_r (int, optional): The target rank. Defaults to 8.
            init_r (int, optional): The initial rank. Defaults to 12.
            tinit (int, optional): The initial temperature. Defaults to 0.
            tfinal (int, optional): The final temperature. Defaults to 0.
            deltaT (int, optional): The temperature change rate. Defaults to 1.
            beta1 (float, optional): The value of beta1 for optimizer. Defaults to 0.85.
            beta2 (float, optional): The value of beta2 for optimizer. Defaults to 0.85.
            orth_reg_weight (float, optional): The weight of the orthogonality regularization. Defaults to 0.5.
            total_step (int, optional): The total number of training steps. Defaults to None.
            rank_pattern (dict, optional): The rank pattern. Defaults to None.
            num_classes (int, optional): The number of classes. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            target_r=target_r,
            init_r=init_r,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2,
            orth_reg_weight=orth_reg_weight,
            total_step=total_step,
            rank_pattern=rank_pattern,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/peft/adalora/llama")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlamaAdaLoraForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlamaAdaLoraForClassification: The initialized LlamaAdaLoraForClassification instance.
        """
        config.set_default_section("core/model/classification/peft/adalora/llama")
        pretrained_name = config.getoption("pretrained_name", "default-llama")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        target_r = config.getoption("target_r", 8)
        init_r = config.getoption("init_r", 12)
        tinit = config.getoption("tinit", 0)
        tfinal = config.getoption("tfinal", 0)
        deltaT = config.getoption("deltaT", 1)
        beta1 = config.getoption("beta1", 0.85)
        beta2 = config.getoption("beta2", 0.85)
        orth_reg_weight = config.getoption("orth_reg_weight", 0.5)
        total_step = config.getoption("total_step", None)
        rank_pattern = config.getoption("rank_pattern", None)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            target_r=target_r,
            init_r=init_r,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2,
            orth_reg_weight=orth_reg_weight,
            total_step=total_step,
            rank_pattern=rank_pattern,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        pretrained_adalora_weight_path = config.getoption(
            "pretrained_adalora_weight_path", None
        )
        if pretrained_adalora_weight_path is not None:
            weight_path.append(pretrained_adalora_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform forward pass of the LlamaAdaLoraForClassification model.

        Args:
            input_ids (torch.Tensor): The input IDs.
            attention_mask (torch.Tensor, optional): The attention mask.
            position_ids (torch.Tensor, optional): The position IDs.

        Returns:
            ClassificationOutputs: The output of the classification task.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/generation/peft/adalora/llama", generation_model_decorator)
class LlamaAdaLoraForGeneration(_LlamaAdaLoraForGeneration):
    """LlamaAdaLora model for generation tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        target_r: Optional[int] = 8,
        init_r: Optional[int] = 12,
        tinit: Optional[int] = 0,
        tfinal: Optional[int] = 0,
        deltaT: Optional[int] = 1,
        beta1: Optional[float] = 0.85,
        beta2: Optional[float] = 0.85,
        orth_reg_weight: Optional[float] = 0.5,
        total_step: Optional[int] = None,
        rank_pattern: Optional[dict] = None,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlamaAdaLoraForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            target_r (int, optional): The target rank. Defaults to 8.
            init_r (int, optional): The initial rank. Defaults to 12.
            tinit (int, optional): The initial temperature. Defaults to 0.
            tfinal (int, optional): The final temperature. Defaults to 0.
            deltaT (int, optional): The temperature change rate. Defaults to 1.
            beta1 (float, optional): The value of beta1 for optimizer. Defaults to 0.85.
            beta2 (float, optional): The value of beta2 for optimizer. Defaults to 0.85.
            orth_reg_weight (float, optional): The weight of the orthogonality regularization. Defaults to 0.5.
            total_step (int, optional): The total number of training steps. Defaults to None.
            rank_pattern (dict, optional): The rank pattern. Defaults to None.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            target_r=target_r,
            init_r=init_r,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2,
            orth_reg_weight=orth_reg_weight,
            total_step=total_step,
            rank_pattern=rank_pattern,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/peft/adalora/llama")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlamaAdaLoraForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlamaAdaLoraForGeneration: The initialized LlamaAdaLoraForGeneration instance.
        """
        config.set_default_section("core/model/generation/peft/adalora/llama")
        pretrained_name = config.getoption("pretrained_name", "default-llama")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        target_r = config.getoption("target_r", 8)
        init_r = config.getoption("init_r", 12)
        tinit = config.getoption("tinit", 0)
        tfinal = config.getoption("tfinal", 0)
        deltaT = config.getoption("deltaT", 1)
        beta1 = config.getoption("beta1", 0.85)
        beta2 = config.getoption("beta2", 0.85)
        orth_reg_weight = config.getoption("orth_reg_weight", 0.5)
        total_step = config.getoption("total_step", None)
        rank_pattern = config.getoption("rank_pattern", None)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            target_r=target_r,
            init_r=init_r,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2,
            orth_reg_weight=orth_reg_weight,
            total_step=total_step,
            rank_pattern=rank_pattern,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        pretrained_adalora_weight_path = config.getoption(
            "pretrained_adalora_weight_path", None
        )
        if pretrained_adalora_weight_path is not None:
            weight_path.append(pretrained_adalora_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform forward pass of the LlamaAdaLoraForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input IDs.
            attention_mask (torch.Tensor, optional): The attention mask.
            position_ids (torch.Tensor, optional): The position IDs.

        Returns:
            GenerationOutputs: The output of the generation task.
        """

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/peft/adalora/llama")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 2,
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
        Generate sequences using the Llama model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 0.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 1.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum generation sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum generation sequence length. Defaults to 48.
            repetition_penalty (float, optional): Repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to prevent repetition. Defaults to 0.
            early_stopping (bool, optional): Whether to perform early stopping. Defaults to True.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().generate(
            input_ids,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return GenerationOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )


@register_model("core/model/classification/peft/lora/llama")
class LlamaLoraForClassification(_LlamaLoraForClassification):
    """LlamaLora model for classification tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlamaLoraForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            lora_r (int, optional): The number of Lora ranks. Defaults to 16.
            lora_alpha (int, optional): The Lora alpha value. Defaults to 32.
            lora_dropout (float, optional): The Lora dropout rate. Defaults to 0.05.
            fan_in_fan_out (bool, optional): Whether to use fan-in/fan-out weight initialization. Defaults to True.
            target_modules (Union[List[str], str], optional): The target modules for Lora regularization. Defaults to ["q_proj", "v_proj"].
            num_classes (int, optional): The number of classes. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/peft/lora/llama")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlamaLoraForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlamaLoraForClassification: The initialized LlamaLoraForClassification instance.
        """
        config.set_default_section("core/model/classification/peft/lora/llama")
        pretrained_name = config.getoption("pretrained_name", "default-llama")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["q_proj", "v_proj"])

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            weight_path.append(pretrained_lora_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform forward pass of the LlamaLoraForClassification model.

        Args:
            input_ids (torch.Tensor): The input IDs.
            attention_mask (torch.Tensor, optional): The attention mask.
            position_ids (torch.Tensor, optional): The position IDs.

        Returns:
            ClassificationOutputs: The output of the classification task.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/generation/peft/lora/llama", generation_model_decorator)
class LlamaLoraForGeneration(_LlamaLoraForGeneration):
    """LlamaLora model for generation tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlamaLoraForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            lora_r (int, optional): The number of Lora ranks. Defaults to 16.
            lora_alpha (int, optional): The Lora alpha value. Defaults to 32.
            lora_dropout (float, optional): The Lora dropout rate. Defaults to 0.05.
            fan_in_fan_out (bool, optional): Whether to use fan-in/fan-out weight initialization. Defaults to True.
            target_modules (Union[List[str], str], optional): The target modules for Lora regularization. Defaults to ["q_proj", "v_proj"].
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/peft/lora/llama")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlamaLoraForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlamaLoraForGeneration: The initialized LlamaLoraForGeneration instance.
        """
        config.set_default_section("core/model/generation/peft/lora/llama")
        pretrained_name = config.getoption("pretrained_name", "default-llama")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["q_proj", "v_proj"])

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            weight_path.append(pretrained_lora_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform forward pass of the LlamaLoraForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input IDs.
            attention_mask (torch.Tensor, optional): The attention mask.
            position_ids (torch.Tensor, optional): The position IDs.

        Returns:
            GenerationOutputs: The output of the generation task.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/peft/lora/llama")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 2,
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
        Generate sequences using the Llama model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 0.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 1.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum generation sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum generation sequence length. Defaults to 48.
            repetition_penalty (float, optional): Repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to prevent repetition. Defaults to 0.
            early_stopping (bool, optional): Whether to perform early stopping. Defaults to True.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().generate(
            input_ids,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            decoder_end_token_id=decoder_end_token_id,
            num_return_sequences=num_return_sequences,
            min_gen_seq_length=min_gen_seq_length,
            max_gen_seq_length=max_gen_seq_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return GenerationOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )
