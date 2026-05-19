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
        pad_token_id: Optional[int] = 151643,
    ) -> GenerationOutputs:
        """
        Generates sequences for the given text and image inputs.

        Passes already-preprocessed ``pixel_values`` (shape ``(B, num_patches, channels)``)
        and ``image_grid_thw`` directly to vLLM via ``mm_processor_kwargs``, bypassing
        vLLM's own image pre-processing pipeline so that the unitorch processor output
        is used as-is (matching the HuggingFace reference implementation).

        Args:
            input_ids (torch.Tensor): Input token ID tensor of shape ``(batch, seq_len)``.
            pixel_values (torch.Tensor, optional): Pre-processed patch tensor of shape
                ``(B, num_patches, channels)`` produced by the unitorch QWenVL processor.
            image_grid_thw (torch.Tensor, optional): Grid metadata tensor of shape
                ``(B, 3)`` containing ``(temporal, height, width)`` patch counts per sample.
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
        from vllm import SamplingParams

        # Qwen3-VL may emit <|im_end|> before the visible answer content, so
        # treating it as a hard stop can collapse the response to an empty
        # string. Use <|endoftext|> as the only stop token here.
        stop_token_ids = [151643]

        sampling_params = SamplingParams(
            n=num_return_sequences,
            max_tokens=max_gen_seq_length,
            min_tokens=min_gen_seq_length,
            temperature=temperature if do_sample else 0.0,
            top_k=top_k if do_sample else -1,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stop_token_ids=stop_token_ids,
        )

        batch_size = input_ids.shape[0]
        inputs = []
        for i in range(batch_size):
            token_ids = [t for t in input_ids[i].tolist() if t != pad_token_id]
            entry: Dict[str, Any] = {"prompt": self._decode_prompt(token_ids)}
            if pixel_values is not None and image_grid_thw is not None:
                grid_thw = image_grid_thw[i]
                if grid_thw.dim() == 1:
                    grid_thw = grid_thw.unsqueeze(0)
                # Pass pre-processed patch tensor directly to the vLLM model via
                # multi_modal_data.  Qwen2VLMultiModalDataParserV2 (registered via
                # @replace in unitorch.models.vllm.modeling_vl) extends vLLM's
                # data parser to accept {"pixel_values", "image_grid_thw"} dicts,
                # routing them to DictEmbeddingItems so the vision encoder receives
                # the exact same patches that the unitorch Qwen2VLImageProcessor
                # produced — bypassing vLLM's own image preprocessing pipeline.
                entry["multi_modal_data"] = {
                    "image": {
                        "pixel_values": pixel_values[i].to(torch.bfloat16),
                        "image_grid_thw": grid_thw,
                    }
                }
            inputs.append(entry)

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        batch_token_ids = [[o.token_ids for o in req.outputs] for req in outputs]
        sequences = _pad_token_ids(batch_token_ids, pad_token_id, max_gen_seq_length)
        return GenerationOutputs(sequences=sequences)
