# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import atexit
import gc
import torch
from typing import Any, Dict, List, Optional, Union
from vllm import LLM, SamplingParams


class VLLMForGeneration:
    """
    Text generation model backed by vLLM offline inference engine.

    Wraps ``vllm.LLM`` for synchronous and asynchronous token generation.
    Accepts tokenized ``input_ids`` tensors (compatible with unitorch-infer)
    and returns token-ID tensors via ``GenerationOutputs``.
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
        trust_remote_code: Optional[bool] = True,
        dtype: Optional[str] = "auto",
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
    ):
        """
        Initializes the vLLM text generation engine.

        Args:
            hf_name_or_folder (str): Path to the HuggingFace model folder.
            tensor_parallel_size (int): Number of GPUs for tensor parallelism. Defaults to 1.
            pipeline_parallel_size (int): Number of GPUs for pipeline parallelism. Defaults to 1.
            gpu_memory_utilization (float): Fraction of GPU memory to reserve for vLLM. Defaults to 0.90.
            max_model_len (int, optional): Maximum sequence length. None uses model default.
            max_num_seqs (int): Maximum number of concurrent sequences. Defaults to 256.
            enable_prefix_caching (bool): Enable automatic KV-cache prefix sharing. Defaults to True.
            trust_remote_code (bool): Allow remote model code execution. Defaults to True.
            dtype (str): Model weight dtype (``"auto"``, ``"float16"``, ``"bfloat16"``). Defaults to ``"auto"``.
            enforce_eager (bool): Disable CUDA graph capture (useful for debugging). Defaults to False.
            quantization (str, optional): Quantization method (``"awq"``, ``"gptq"``, etc.).
        """
        kwargs = dict(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enforce_eager=enforce_eager,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            kwargs["quantization"] = quantization

        self.llm = LLM(model=hf_name_or_folder, **kwargs)
        atexit.register(self.shutdown)

    def cuda(self, device=None):
        # vLLM manages GPU placement internally at engine init time.
        # This method exists for interface compatibility with nn.Module-based models.
        return self

    def eval(self):
        # vLLM is always in inference mode; this is a no-op for interface compatibility.
        return self

    def train(self, mode=True):
        # vLLM does not support training mode; this is a no-op for interface compatibility.
        return self

    def from_checkpoint(self, ckpt_dir, **kwargs):
        # vLLM loads weights at engine init time from hf_name_or_folder.
        # Post-init checkpoint loading is not supported and is silently ignored.
        pass

    def shutdown(self):
        """Shutdown the vLLM engine and release GPU/IPC resources held by worker processes."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return
        # Try shutdown from outermost layer inward so the highest-level
        # teardown (which joins worker processes) runs first.
        for target in (
            engine,
            getattr(engine, "engine_core", None),
        ):
            if target is not None and hasattr(target, "shutdown"):
                try:
                    target.shutdown()
                    break
                except Exception:
                    pass
        # Drop the strong reference so Python's GC can reclaim vLLM's internal
        # shared-memory / semaphore objects before the resource_tracker checks.
        self.llm = None
        gc.collect()

    def __del__(self):
        # Belt-and-suspenders: catches cases where atexit never fires
        # (e.g. SIGKILL, early GC before interpreter teardown).
        self.shutdown()

    def generate(
        self,
        input_ids: torch.Tensor,
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
    ) -> List[List[List[int]]]:
        """
        Generates token sequences for the given input_ids.

        Args:
            input_ids (torch.Tensor): Input token ID tensor of shape ``(batch, seq_len)``.
            max_gen_seq_length (int): Maximum number of new tokens to generate. Defaults to 512.
            min_gen_seq_length (int): Minimum number of new tokens to generate. Defaults to 0.
            num_return_sequences (int): Number of completions per prompt. Defaults to 1.
            num_beams (int): Beam search width (used when ``use_beam_search=True``). Defaults to 1.
            do_sample (bool): Enable sampling; when False uses greedy/beam decoding. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p (nucleus) sampling parameter. Defaults to 1.0.
            repetition_penalty (float): Penalty for token repetition. Defaults to 1.0.
            length_penalty (float): Exponential length penalty for beam search. Defaults to 1.0.
            stop (str or List[str], optional): Stop strings that terminate generation.
            use_beam_search (bool): Use beam search instead of sampling. Defaults to False.

        Returns:
            List[List[List[int]]]: Generated token ID sequences,
            shape ``[batch][num_return_sequences][seq_len]``.
        """
        # Always stop at <|im_end|> (151645) and <|endoftext|> (151643) so that
        # vLLM does not generate past the model's answer turn into reasoning/thinking text.
        stop_token_ids = [151643, 151645]

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

        # Convert tensor rows to prompt_token_ids format (strips padding tokens)
        prompts = [
            {"prompt_token_ids": [t for t in row.tolist() if t != pad_token_id]}
            for row in input_ids
        ]

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        return [[o.token_ids for o in req.outputs] for req in outputs]

    async def async_generate(
        self,
        input_ids: torch.Tensor,
        max_gen_seq_length: Optional[int] = 512,
        min_gen_seq_length: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> List[List[int]]:
        """
        Asynchronously generates token sequences for a single-row input_ids tensor.

        Args:
            input_ids (torch.Tensor): Input token ID tensor of shape ``(1, seq_len)`` or ``(seq_len,)``.
            max_gen_seq_length (int): Maximum tokens to generate. Defaults to 512.
            min_gen_seq_length (int): Minimum tokens to generate. Defaults to 0.
            num_return_sequences (int): Number of completions. Defaults to 1.
            do_sample (bool): Enable sampling. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Top-k sampling. Defaults to 50.
            top_p (float): Top-p sampling. Defaults to 1.0.
            repetition_penalty (float): Repetition penalty. Defaults to 1.0.
            stop (str or List[str], optional): Stop strings.

        Returns:
            List[List[int]]: Generated token ID sequences for the single prompt.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        results = self.generate(
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
        )
        return results[0]
