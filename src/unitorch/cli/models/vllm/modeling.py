# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Dict, List, Optional, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.vllm import VLLMForGeneration as _VLLMForGeneration
from unitorch.cli import config_defaults_init, register_model
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import GenerationOutputs
from unitorch.cli.models.vllm import pretrained_vllm_infos


def _pad_token_ids(
    batch: List[List[List[int]]],
    pad_id: int,
    max_len: int,
) -> torch.Tensor:
    """Pad ``batch[B][N][T]`` token ID lists into a ``(B, N, max_len)`` tensor."""
    B = len(batch)
    N = max(len(seqs) for seqs in batch)
    result = torch.full((B, N, max_len), pad_id, dtype=torch.long)
    for b, seqs in enumerate(batch):
        for n, ids in enumerate(seqs):
            length = min(len(ids), max_len)
            result[b, n, :length] = torch.tensor(ids[:length], dtype=torch.long)
    if N == 1:
        result = result.squeeze(1)
    return result


@register_model("core/model/vllm/generation/qwen3")
class QWen3VLLMForGeneration(_VLLMForGeneration):
    """
    QWen3 text generation model using the vLLM inference engine.

    Uses vLLM's offline batch engine for high-throughput inference.
    Accepts tokenized ``input_ids`` tensors and returns ``GenerationOutputs``
    compatible with ``unitorch-infer``.
    """

    def __init__(
        self,
        hf_name_or_folder: str,
        tensor_parallel_size: Optional[int] = 1,
        pipeline_parallel_size: Optional[int] = 1,
        gpu_memory_utilization: Optional[float] = 0.90,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = 256,
        enable_prefix_caching: Optional[bool] = True,
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
            enable_prefix_caching=enable_prefix_caching,
            dtype=dtype,
            enforce_eager=enforce_eager,
            quantization=quantization,
        )

    @classmethod
    @config_defaults_init("core/model/vllm/generation/qwen3")
    def from_config(cls, config, **kwargs):
        config.set_default_section("core/model/vllm/generation/qwen3")
        pretrained_name = config.getoption("pretrained_name", "qwen3-4b-thinking")

        hf_name_or_folder = config.getoption("hf_name_or_folder", None)
        hf_name_or_folder = pop_value(
            hf_name_or_folder,
            nested_dict_value(pretrained_vllm_infos, pretrained_name, "hf_pretrained_name"),
        )

        tensor_parallel_size = config.getoption("tensor_parallel_size", 1)
        pipeline_parallel_size = config.getoption("pipeline_parallel_size", 1)
        gpu_memory_utilization = config.getoption("gpu_memory_utilization", 0.90)
        max_model_len = config.getoption("max_model_len", None)
        max_num_seqs = config.getoption("max_num_seqs", 256)
        enable_prefix_caching = config.getoption("enable_prefix_caching", True)
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
            enable_prefix_caching=enable_prefix_caching,
            dtype=dtype,
            enforce_eager=enforce_eager,
            quantization=quantization,
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        max_gen_seq_length: Optional[int] = 512,
        min_gen_seq_length: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        num_beams: Optional[int] = 1,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        pad_token_id: Optional[int] = 151643,
    ) -> GenerationOutputs:
        """
        Generates sequences for the given input token IDs.

        Args:
            input_ids (torch.Tensor): Input token ID tensor of shape ``(batch, seq_len)``.
            max_gen_seq_length (int): Maximum tokens to generate. Defaults to 512.
            min_gen_seq_length (int): Minimum tokens to generate. Defaults to 0.
            num_return_sequences (int): Completions per prompt. Defaults to 1.
            num_beams (int): Beam search width. Defaults to 1.
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
            max_gen_seq_length=max_gen_seq_length,
            min_gen_seq_length=min_gen_seq_length,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
            pad_token_id=pad_token_id,
        )
        sequences = _pad_token_ids(batch_token_ids, pad_token_id, max_gen_seq_length)
        return GenerationOutputs(sequences=sequences)
