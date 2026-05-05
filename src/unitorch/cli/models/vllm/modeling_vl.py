# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.vllm import VLLMVLForGeneration as _VLLMVLForGeneration
from unitorch.cli import config_defaults_init, register_model
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import GenerationOutputs
from unitorch.cli.models.vllm import pretrained_vllm_infos
from unitorch.cli.models.vllm.modeling import _pad_token_ids


@register_model("core/model/vllm/generation/qwen3_vl")
class QWen3VLVLLMForGeneration(_VLLMVLForGeneration):
    """
    QWen3-VL vision-language generation model using the vLLM inference engine.

    Uses vLLM's multimodal offline batch engine for high-throughput inference
    over text and image inputs. Accepts tokenized ``input_ids`` tensors and
    pixel-values tensors (or raw ``PIL.Image``) and returns ``GenerationOutputs``
    compatible with ``unitorch-infer``.
    """

    def __init__(
        self,
        hf_name_or_folder: str,
        tensor_parallel_size: Optional[int] = 1,
        pipeline_parallel_size: Optional[int] = 1,
        gpu_memory_utilization: Optional[float] = 0.90,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = 128,
        max_num_images: Optional[int] = 8,
        enable_prefix_caching: Optional[bool] = False,
        dtype: Optional[str] = "auto",
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
    ):
        super().__init__(
            hf_name_or_folder=hf_name_or_folder,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_images=max_num_images,
            enable_prefix_caching=enable_prefix_caching,
            dtype=dtype,
            enforce_eager=enforce_eager,
            quantization=quantization,
        )

    @classmethod
    @config_defaults_init("core/model/vllm/generation/qwen3_vl")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/vllm/generation/qwen3_vl")
        pretrained_name = config.getoption("pretrained_name", "qwen3-vl-2b-instruct")

        hf_name_or_folder = config.getoption("hf_name_or_folder", None)
        hf_name_or_folder = pop_value(
            hf_name_or_folder,
            nested_dict_value(pretrained_vllm_infos, pretrained_name, "hf_pretrained_name"),
        )

        tensor_parallel_size = config.getoption("tensor_parallel_size", 1)
        pipeline_parallel_size = config.getoption("pipeline_parallel_size", 1)
        gpu_memory_utilization = config.getoption("gpu_memory_utilization", 0.90)
        max_model_len = config.getoption("max_model_len", None)
        max_num_seqs = config.getoption("max_num_seqs", 128)
        max_num_images = config.getoption("max_num_images", 8)
        enable_prefix_caching = config.getoption("enable_prefix_caching", False)
        dtype = config.getoption("dtype", "auto")
        enforce_eager = config.getoption("enforce_eager", False)
        quantization = config.getoption("quantization", None)

        return cls(
            hf_name_or_folder=hf_name_or_folder,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_images=max_num_images,
            enable_prefix_caching=enable_prefix_caching,
            dtype=dtype,
            enforce_eager=enforce_eager,
            quantization=quantization,
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_gen_seq_length: Optional[int] = 512,
        min_gen_seq_length: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        pad_token_id: Optional[int] = 0,
    ) -> GenerationOutputs:
        """
        Generates sequences for the given text and image inputs.

        Args:
            input_ids (torch.Tensor): Input token ID tensor of shape ``(batch, seq_len)``.
            pixel_values (torch.Tensor, optional): Pixel-values tensor ``(B, C, H, W)``
                or ``(C, H, W)``. Passed to vLLM as PIL images after conversion.
            image_grid_thw (torch.Tensor, optional): Unused; accepted for interface
                compatibility with the standard QWen3-VL generate signature.
            max_gen_seq_length (int): Maximum tokens to generate. Defaults to 512.
            min_gen_seq_length (int): Minimum tokens to generate. Defaults to 0.
            num_return_sequences (int): Completions per prompt. Defaults to 1.
            do_sample (bool): Enable sampling. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Top-k sampling. Defaults to 50.
            top_p (float): Top-p sampling. Defaults to 1.0.
            repetition_penalty (float): Repetition penalty. Defaults to 1.0.
            stop (str or List[str], optional): Stop strings.
            pad_token_id (int): Token ID used for padding. Defaults to 0.

        Returns:
            GenerationOutputs: Sequences tensor of shape ``(batch, num_return_sequences, max_gen_seq_length)``.
        """
        batch_token_ids = super().generate(
            input_ids=input_ids,
            images=pixel_values,
            max_gen_seq_length=max_gen_seq_length,
            min_gen_seq_length=min_gen_seq_length,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )
        sequences = _pad_token_ids(batch_token_ids, pad_token_id, max_gen_seq_length)
        return GenerationOutputs(sequences=sequences)
