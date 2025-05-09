# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value, is_bfloat16_available
from unitorch.models.llava import (
    LlavaMistralClipForClassification as _LlavaMistralClipForClassification,
    LlavaMistralClipForGeneration as _LlavaMistralClipForGeneration,
    LlavaLlamaSiglipForGeneration as _LlavaLlamaSiglipForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs
from unitorch.cli.models.llava import (
    pretrained_llava_infos,
    pretrained_llava_extensions_infos,
)


@register_model("core/model/classification/llava/mistral_clip")
class LlavaMistralClipForClassification(_LlavaMistralClipForClassification):
    """LlavaMistralClip model for classification tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        image_token_index: Optional[int] = 32000,
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
        freeze_vision_encoder: Optional[bool] = True,
        freeze_multi_modal_projector: Optional[bool] = True,
        freeze_llm_encoder: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlavaMistralClipForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            num_classes (int, optional): The number of classes for classification. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            image_token_index=image_token_index,
            num_classes=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_llm_encoder=freeze_llm_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/classification/llava/mistral_clip")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlavaMistralClipForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlavaMistralClipForClassification: An instance of LlavaMistralClipForClassification.
        """
        config.set_default_section("core/model/classification/llava/mistral_clip")
        pretrained_name = config.getoption(
            "pretrained_name", "llava-v1.6-mistral-7b-hf"
        )
        pretrained_lora_name = config.getoption(
            "pretrained_lora_name", "llava-v1.6-mistral-7b-lora"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)
        image_token_index = config.getoption("image_token_index", 32000)
        num_classes = config.getoption("num_classes", 1)
        hidden_dropout_prob = config.getoption("hidden_dropout_prob", 0.1)
        freeze_vision_encoder = config.getoption("freeze_vision_encoder", True)
        freeze_multi_modal_projector = config.getoption(
            "freeze_multi_modal_projector", True
        )
        freeze_llm_encoder = config.getoption("freeze_llm_encoder", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            image_token_index=image_token_index,
            num_classes=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_llm_encoder=freeze_llm_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = pop_value(
            pretrained_lora_weight_path,
            nested_dict_value(pretrained_llava_extensions_infos, pretrained_lora_name),
            check_none=False,
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if lora_weight_path is not None:
            inst.load_lora_weights(
                lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass on the LlavaMistralClipForClassification model.

        Args:
            input_ids (torch.Tensor): The input tensor containing the input IDs.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor. Defaults to None.

        Returns:
            ClassificationOutputs: The output of the classification model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("core/model/generation/llava/mistral_clip", generation_model_decorator)
class LlavaMistralClipForGeneration(_LlavaMistralClipForGeneration):
    """LlavaMistralClip model for generation tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        image_token_index: Optional[int] = 32000,
        freeze_vision_encoder: Optional[bool] = True,
        freeze_multi_modal_projector: Optional[bool] = False,
        freeze_llm_encoder: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlavaMistralClipForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            image_token_index=image_token_index,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_llm_encoder=freeze_llm_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/llava/mistral_clip")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlavaMistralClipForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlavaMistralClipForGeneration: An instance of LlavaMistralClipForGeneration.
        """
        config.set_default_section("core/model/generation/llava/mistral_clip")
        pretrained_name = config.getoption(
            "pretrained_name", "llava-v1.6-mistral-7b-hf"
        )
        pretrained_lora_name = config.getoption(
            "pretrained_lora_name", "llava-v1.6-mistral-7b-lora"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)
        image_token_index = config.getoption("image_token_index", 32000)
        freeze_vision_encoder = config.getoption("freeze_vision_encoder", True)
        freeze_multi_modal_projector = config.getoption(
            "freeze_multi_modal_projector", False
        )
        freeze_llm_encoder = config.getoption("freeze_llm_encoder", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            image_token_index=image_token_index,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_llm_encoder=freeze_llm_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = pop_value(
            pretrained_lora_weight_path,
            nested_dict_value(pretrained_llava_extensions_infos, pretrained_lora_name),
            check_none=False,
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if lora_weight_path is not None:
            inst.load_lora_weights(
                lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass on the LlavaMistralClipForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input tensor containing the input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor. Defaults to None.

        Returns:
            GenerationOutputs: The output of the generation model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/llava/mistral_clip")
    @torch.no_grad()
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        Generate sequences using the LlavaMistralClip model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 1.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 2.
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
            pixel_values=pixel_values,
            attention_mask=attention_mask,
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


@register_model("core/model/generation/llava/llama_siglip", generation_model_decorator)
class LlavaLlamaSiglipForGeneration(_LlavaLlamaSiglipForGeneration):
    """LlavaMistralClip model for generation tasks."""

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        image_token_index: Optional[int] = 128077,
        freeze_vision_encoder: Optional[bool] = True,
        freeze_multi_modal_projector: Optional[bool] = False,
        freeze_llm_encoder: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the LlavaMistralClipForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            quant_config_path=quant_config_path,
            image_token_index=image_token_index,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_llm_encoder=freeze_llm_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/llava/llama_siglip")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of LlavaMistralClipForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LlavaMistralClipForGeneration: An instance of LlavaMistralClipForGeneration.
        """
        config.set_default_section("core/model/generation/llava/llama_siglip")
        pretrained_name = config.getoption("pretrained_name", "llava-v1.6-joycaption-2")
        pretrained_lora_name = config.getoption(
            "pretrained_lora_name", "llava-v1.6-joycaption-2-lora"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)
        image_token_index = config.getoption("image_token_index", 128077)
        freeze_vision_encoder = config.getoption("freeze_vision_encoder", True)
        freeze_multi_modal_projector = config.getoption(
            "freeze_multi_modal_projector", False
        )
        freeze_llm_encoder = config.getoption("freeze_llm_encoder", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            image_token_index=image_token_index,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_llm_encoder=freeze_llm_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_llava_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = pop_value(
            pretrained_lora_weight_path,
            nested_dict_value(pretrained_llava_extensions_infos, pretrained_lora_name),
            check_none=False,
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if lora_weight_path is not None:
            inst.load_lora_weights(
                lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    # @autocast(
    #     device_type=("cuda" if torch.cuda.is_available() else "cpu"),
    #     dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    # )
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass on the LlavaMistralClipForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input tensor containing the input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor. Defaults to None.

        Returns:
            GenerationOutputs: The output of the generation model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/llava/llama_siglip")
    @torch.no_grad()
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 128000,
        decoder_end_token_id: Optional[Union[int, List[int]]] = [
            128001,
            128008,
            128009,
        ],
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
        Generate sequences using the LlavaMistralClip model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 1.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 2.
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
            pixel_values=pixel_values,
            attention_mask=attention_mask,
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
