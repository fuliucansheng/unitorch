# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import gc
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import cached_path, config_defaults_init
from unitorch.cli import register_fastapi
from unitorch.cli import Config, GenericFastAPI
from unitorch.cli.models.vllm import pretrained_vllm_infos
from unitorch.cli.models.vllm.modeling import QWen3VLLMForGeneration


@register_fastapi("core/fastapi/vllm/qwen3")
class QWen3VLLMFastAPI(GenericFastAPI):
    """
    FastAPI service for QWen3 text generation powered by vLLM.

    Exposes ``/generate``, ``/status``, ``/start``, and ``/stop`` endpoints
    under a configurable router prefix (default ``/core/fastapi/vllm/qwen3``).
    """

    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/vllm/qwen3")
        router = config.getoption("router", "/core/fastapi/vllm/qwen3")
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self, pretrained_name: str = "qwen3-4b-thinking"):
        """
        Loads and starts the vLLM QWen3 engine.

        Args:
            pretrained_name (str): Pretrained model name to load. Defaults to ``"qwen3-4b-thinking"``.
        """
        pretrained_name_or_path = nested_dict_value(
            pretrained_vllm_infos, pretrained_name, "pretrained_name_or_path"
        )
        self.config.set_default_section("core/fastapi/vllm/qwen3")
        if pretrained_name_or_path is not None:
            self.config.set(
                "core/fastapi/vllm/qwen3", "pretrained_name", pretrained_name
            )
        self._pipe = QWen3VLLMForGeneration.from_config(
            self.config,
            pretrained_name=pretrained_name,
        )
        return "start success"

    def stop(self):
        """
        Stops and unloads the vLLM engine, releasing GPU memory.
        """
        del self._pipe
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        self._pipe = None
        return "stop success"

    def status(self):
        """Returns ``"running"`` if the engine is loaded, otherwise ``"stopped"``."""
        return "running" if self._pipe is not None else "stopped"

    async def generate(
        self,
        text: str,
        use_chat_template: Optional[bool] = True,
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
    ):
        """
        Generates a text completion for the given prompt.

        Args:
            text (str): Input prompt or JSON-encoded message list (when ``use_chat_template=True``).
            use_chat_template (bool): Apply chat template formatting. Defaults to True.
            max_gen_seq_length (int): Maximum tokens to generate. Defaults to 512.
            min_gen_seq_length (int): Minimum tokens to generate. Defaults to 0.
            num_return_sequences (int): Number of completions to return. Defaults to 1.
            num_beams (int): Beam search width. Defaults to 1.
            do_sample (bool): Enable sampling-based decoding. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Top-k sampling. Defaults to 50.
            top_p (float): Top-p (nucleus) sampling. Defaults to 1.0.
            repetition_penalty (float): Repetition penalty. Defaults to 1.0.
            stop (str or List[str], optional): Stop string(s) to end generation.

        Returns:
            str or List[str]: Generated text. Single string when ``num_return_sequences=1``.
        """
        assert self._pipe is not None, "Service not started. Call /start first."
        processor = self._pipe.processor
        prompt = (
            processor.chat_template(messages=json.loads(text))
            if use_chat_template
            else text
        )
        inputs = processor.generation_inputs(text=prompt)
        import torch

        input_ids = inputs.input_ids.unsqueeze(0)
        async with self._lock:
            outputs = self._pipe.generate(
                input_ids=input_ids,
                max_gen_seq_length=max_gen_seq_length,
                min_gen_seq_length=min_gen_seq_length,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
        decoded = processor.detokenize(sequences=outputs.sequences)
        sequences = decoded[0]
        return sequences[0] if num_return_sequences == 1 else sequences
