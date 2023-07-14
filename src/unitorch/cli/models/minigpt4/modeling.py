# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.minigpt4 import (
    MiniGPT4Blip2LlamaForGeneration as _MiniGPT4Blip2LlamaForGeneration,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.minigpt4 import pretrained_minigpt4_infos


@register_model("core/model/generation/minigpt4", generation_model_decorator)
class MiniGPT4Blip2LlamaForGeneration(_MiniGPT4Blip2LlamaForGeneration):
    """
    MiniGPT4Blip2LlamaForGeneration is a class for generating sequences using the MiniGPT4 model with Blip2 and Llama.
    It inherits from the _MiniGPT4Blip2LlamaForGeneration class.
    """

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        pad_token_id: Optional[int] = 0,
        freeze_vision_model: Optional[bool] = True,
        freeze_qformer_model: Optional[bool] = True,
        freeze_llama_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initializes a MiniGPT4Blip2LlamaForGeneration instance.

        Args:
            blip2_config_path (str): The file path to the Blip2 configuration.
            llama_config_path (str): The file path to the Llama configuration.
            pad_token_id (int, optional): The ID of the padding token. Defaults to 0.
            freeze_vision_model (bool, optional): Whether to freeze the vision model. Defaults to True.
            freeze_qformer_model (bool, optional): Whether to freeze the query transformer model. Defaults to True.
            freeze_llama_model (bool, optional): Whether to freeze the Llama model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            blip2_config_path=blip2_config_path,
            llama_config_path=llama_config_path,
            pad_token_id=pad_token_id,
            freeze_vision_model=freeze_vision_model,
            freeze_qformer_model=freeze_qformer_model,
            freeze_llama_model=freeze_llama_model,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    @add_default_section_for_init("core/model/generation/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a MiniGPT4Blip2LlamaForGeneration instance from a core configuration.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            MiniGPT4Blip2LlamaForGeneration: The created instance.
        """
        config.set_default_section("core/model/generation/minigpt4")
        pretrained_name = config.getoption("pretrained_name", "default-minigpt4")

        blip2_config_path = config.getoption("blip2_config_path", None)
        blip2_config_path = pop_value(
            blip2_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "blip2_config_path"
            ),
        )
        blip2_config_path = cached_path(blip2_config_path)

        llama_config_path = config.getoption("llama_config_path", None)
        llama_config_path = pop_value(
            llama_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "llama_config_path"
            ),
        )
        llama_config_path = cached_path(llama_config_path)

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        freeze_qformer_model = config.getoption("freeze_qformer_model", True)
        freeze_llama_model = config.getoption("freeze_llama_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            blip2_config_path,
            llama_config_path,
            freeze_vision_model=freeze_vision_model,
            freeze_qformer_model=freeze_qformer_model,
            freeze_llama_model=freeze_llama_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast()
    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            decoder_input_ids (torch.Tensor): The input IDs for the decoder tokens.
            prefix_attention_mask (torch.Tensor, optional): The attention mask for the prefix tokens.
                Defaults to None.
            suffix_attention_mask (torch.Tensor, optional): The attention mask for the suffix tokens.
                Defaults to None.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder tokens.
                Defaults to None.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().forward(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            decoder_input_ids=decoder_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return GenerationOutputs(sequences=outputs)

    @add_default_section_for_function("core/model/generation/minigpt4")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
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
        Generates sequences using the model.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 1.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s).
                Defaults to 2.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): The minimum generated sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to 48.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): The size of the n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to perform early stopping. Defaults to True.
            length_penalty (float, optional): The length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): The number of beam groups for diverse beam search.
                Defaults to 1.
            diversity_penalty (float, optional): The diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling instead of beam search. Defaults to False.
            temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
            top_k (int, optional): The value for top-k sampling. Defaults to 50.
            top_p (float, optional): The value for top-p (nucleus) sampling. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = super().generate(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
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
