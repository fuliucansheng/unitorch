# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import atexit
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from vllm import LLM, SamplingParams

class VLLMVLForGeneration:
    """
    Vision-language generation model backed by vLLM offline inference engine.

    Wraps ``vllm.LLM`` for multimodal (text + image) generation supporting
    both single and multi-image inputs via the vLLM multimodal data API.
    Accepts tokenized ``input_ids`` tensors and pixel-values tensors
    (compatible with unitorch-infer) in addition to raw ``PIL.Image`` inputs.
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
        trust_remote_code: Optional[bool] = True,
        dtype: Optional[str] = "auto",
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
    ):
        """
        Initializes the vLLM vision-language generation engine.

        Args:
            hf_name_or_folder (str): Path to the HuggingFace model folder.
            tensor_parallel_size (int): Number of GPUs for tensor parallelism. Defaults to 1.
            pipeline_parallel_size (int): Number of GPUs for pipeline parallelism. Defaults to 1.
            gpu_memory_utilization (float): Fraction of GPU memory to reserve. Defaults to 0.90.
            max_model_len (int, optional): Maximum total sequence length. None uses model default.
            max_num_seqs (int): Maximum concurrent sequences. Defaults to 128.
            max_num_images (int): Maximum images per request (vLLM limit_mm_per_prompt). Defaults to 8.
            enable_prefix_caching (bool): Enable KV-cache prefix sharing. Defaults to False.
            trust_remote_code (bool): Allow remote model code. Defaults to True.
            dtype (str): Weight dtype. Defaults to ``"auto"``.
            enforce_eager (bool): Disable CUDA graph capture. Defaults to False.
            quantization (str, optional): Quantization method.
        """
        kwargs = dict(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": max_num_images},
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enforce_eager=enforce_eager,
            enable_mm_embeds=True,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            kwargs["quantization"] = quantization

        self.llm = LLM(model=hf_name_or_folder, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
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
        """Shutdown the vLLM engine and release GPU memory held by worker processes."""
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass

    def _decode_prompt(self, token_ids: List[int]) -> str:
        """
        Decode prompt token IDs back to the multimodal prompt string expected by vLLM.

        The unitorch processor expands a single ``<|image_pad|>`` / ``<|video_pad|>``
        placeholder into a long run of repeated special tokens based on the visual
        grid size. vLLM expects the *unexpanded* chat-template string and performs
        the multimodal expansion internally, so we collapse those runs before
        decoding the prompt text.
        """
        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        video_token_id = getattr(self.tokenizer, "video_token_id", None)
        if video_token_id is None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        mm_token_ids = {
            token_id
            for token_id in (image_token_id, video_token_id)
            if token_id is not None
        }
        if mm_token_ids:
            collapsed = []
            for token_id in token_ids:
                if (
                    token_id in mm_token_ids
                    and collapsed
                    and collapsed[-1] == token_id
                ):
                    continue
                collapsed.append(token_id)
            token_ids = collapsed

        try:
            return self.tokenizer.batch_decode(
                [token_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
        except TypeError:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _normalize_images(
        self,
        images: Optional[Union[torch.Tensor, Image.Image, List]],
        batch_size: int,
    ) -> Optional[List[Optional[List[Image.Image]]]]:
        """
        Normalize images input to ``List[Optional[List[PIL.Image]]]`` of length ``batch_size``.

        Accepts:
        - ``None``: no images for any prompt.
        - ``torch.Tensor``: shape ``(B, C, H, W)`` or ``(C, H, W)`` pixel-values tensor.
        - ``PIL.Image``: single image shared across all prompts.
        - ``List[PIL.Image]``: one image per prompt.
        - ``List[torch.Tensor]``: one pixel-values tensor per prompt.
        - ``List[List[PIL.Image or torch.Tensor]]``: multiple images per prompt.
        """
        if images is None:
            return None

        # torch.Tensor pixel_values batch (B, C, H, W) or single (C, H, W)
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                return [[images[i]] for i in range(images.shape[0])]
            elif images.dim() == 3:
                return [[images]] * batch_size
            else:
                raise ValueError(f"Unexpected pixel_values shape: {images.shape}")

        if isinstance(images, Image.Image):
            return [[images]] * batch_size

        # List input
        if isinstance(images, list):
            result = []
            for item in images:
                if isinstance(item, (Image.Image, torch.Tensor)):
                    result.append([item])
                elif isinstance(item, list):
                    result.append(item)
                else:
                    raise ValueError(f"Unexpected image type: {type(item)}")
            return result

        raise ValueError(f"Unsupported images type: {type(images)}")

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[Union[torch.Tensor, Image.Image, List]] = None,
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
        Generates token sequences for the given text and image inputs.

        Args:
            input_ids (torch.Tensor): Input token ID tensor of shape ``(batch, seq_len)``.
            images: Input image(s). Accepts:

                - ``None`` — text-only generation.
                - ``torch.Tensor`` — pixel-values tensor ``(B, C, H, W)`` or ``(C, H, W)``.
                - ``PIL.Image`` — single image shared across all prompts.
                - ``List[PIL.Image]`` — one image per prompt.
                - ``List[torch.Tensor]`` — one pixel-values tensor per prompt.
                - ``List[List[PIL.Image or torch.Tensor]]`` — multiple images per prompt.
            max_gen_seq_length (int): Maximum new tokens. Defaults to 512.
            min_gen_seq_length (int): Minimum new tokens. Defaults to 0.
            num_return_sequences (int): Completions per prompt. Defaults to 1.
            do_sample (bool): Enable sampling. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Top-k sampling. Defaults to 50.
            top_p (float): Top-p sampling. Defaults to 1.0.
            repetition_penalty (float): Repetition penalty. Defaults to 1.0.
            stop (str or List[str], optional): Stop strings.

        Returns:
            List[List[List[int]]]: Generated token ID sequences,
            shape ``[batch][num_return_sequences][seq_len]``.
        """
        # Qwen3-VL may legitimately emit <|im_end|> as part of the assistant turn
        # boundary, so stopping on it can truncate the entire answer. Keep
        # <|endoftext|> as the only hard stop token.
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
        normalized_images = self._normalize_images(images, batch_size)

        inputs = []
        for i, row in enumerate(input_ids):
            token_ids = [t for t in row.tolist() if t != pad_token_id]
            entry: Dict[str, Any]
            if normalized_images is not None and normalized_images[i]:
                imgs = normalized_images[i]
                entry = {"prompt": self._decode_prompt(token_ids)}
                entry["multi_modal_data"] = {
                    "image": imgs[0] if len(imgs) == 1 else imgs
                }
            else:
                entry = {"prompt_token_ids": token_ids}
            inputs.append(entry)

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return [[o.token_ids for o in req.outputs] for req in outputs]
        
    async def async_generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[Union[torch.Tensor, Image.Image, List]] = None,
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
        Asynchronously generates token sequences for a single-row input.

        Args:
            input_ids (torch.Tensor): Token ID tensor of shape ``(1, seq_len)`` or ``(seq_len,)``.
            images: Optional image(s) for the single prompt (same formats as ``generate``).
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
            images=images,
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
