# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import json
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile, File
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import cached_path, config_defaults_init
from unitorch.cli import register_fastapi
from unitorch.cli import Config, GenericFastAPI
from unitorch.cli import PipelineReplicaPool
from unitorch.cli.models.vllm import pretrained_vllm_infos
from unitorch.cli.models.vllm.modeling_vl import QWen3VLVLLMForGeneration


@register_fastapi("core/fastapi/vllm/qwen3_vl")
class QWen3VLVLLMFastAPI(GenericFastAPI):
    """
    FastAPI service for QWen3-VL vision-language generation powered by vLLM.

    Exposes ``/generate``, ``/status``, ``/start``, and ``/stop`` endpoints
    under a configurable router prefix (default ``/core/fastapi/vllm/qwen3_vl``).
    Accepts both text-only and multimodal (text + image) generation requests.
    """

    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/vllm/qwen3_vl")
        self._section = "core/fastapi/vllm/qwen3_vl"
        router = config.getoption("router", "/core/fastapi/vllm/qwen3_vl")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self, pretrained_name: str = "qwen3-vl-2b-instruct"):
        """
        Loads and starts the vLLM QWen3-VL multimodal engine.

        Args:
            pretrained_name (str): Pretrained model name to load. Defaults to ``"qwen3-vl-2b-instruct"``.
        """
        self.config.set_default_section(self._section)
        self.config.set(
            self._section, "pretrained_name", pretrained_name
        )
        num_replicas = int(
            self.config.getdefault(self._section, "pipeline_num_replicas", 1)
        )
        devices = self.config.getdefault(
            self._section, "pipeline_replica_devices", "cpu"
        )
        lock = self.config.getdefault(
            self._section, "pipeline_replica_lock", True
        )
        if devices is None:
            devices = []
        if isinstance(devices, str):
            devices = [devices] * num_replicas
        pipelines = [
            QWen3VLVLLMForGeneration.from_config(
                self.config,
                pretrained_name=pretrained_name,
            )
            for _ in range(num_replicas)
        ]
        for pipe, device in zip(pipelines, devices):
            if device is not None and hasattr(pipe, "to"):
                pipe.to(device)
        self._pipes = PipelineReplicaPool(pipelines, lock=lock)
        return "start success"

    def stop(self):
        """
        Stops and unloads the vLLM engine, releasing GPU memory.
        """
        if self._pipes is not None:
            self._pipes.close()
        self._pipes = None
        return "stop success"

    def status(self):
        """Returns ``"running"`` if the engine is loaded, otherwise ``"stopped"``."""
        return "running" if self._pipes is not None else "stopped"

    async def generate(
        self,
        text: str,
        image: Optional[UploadFile] = File(default=None),
        use_chat_template: Optional[bool] = True,
        max_gen_seq_length: Optional[int] = 512,
        min_gen_seq_length: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        """
        Generates a text completion for the given prompt and optional image.

        Args:
            text (str): Input prompt or JSON-encoded message list (when ``use_chat_template=True``).
            image (UploadFile, optional): Uploaded image file for multimodal generation.
            use_chat_template (bool): Apply chat template formatting. Defaults to True.
            max_gen_seq_length (int): Maximum tokens to generate. Defaults to 512.
            min_gen_seq_length (int): Minimum tokens to generate. Defaults to 0.
            num_return_sequences (int): Number of completions to return. Defaults to 1.
            do_sample (bool): Enable sampling-based decoding. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Top-k sampling. Defaults to 50.
            top_p (float): Top-p (nucleus) sampling. Defaults to 1.0.
            repetition_penalty (float): Repetition penalty. Defaults to 1.0.
            stop (str or List[str], optional): Stop string(s).

        Returns:
            str or List[str]: Generated text. Single string when ``num_return_sequences=1``.
        """
        assert self._pipes is not None, "Service not started. Call /start first."

        pil_image = None
        if image is not None:
            content = await image.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")

        pipe = self._pipes.acquire()
        try:
            processor = pipe.processor
            prompt = (
                processor.chat_template(messages=json.loads(text))
                if use_chat_template
                else text
            )
            inputs = processor.generation_inputs(
                text=prompt,
                images=[pil_image] if pil_image is not None else [],
            )
            input_ids = inputs.input_ids.unsqueeze(0)
            pixel_values = (
                inputs.pixel_values.unsqueeze(0) if pil_image is not None else None
            )
            image_grid_thw = inputs.image_grid_thw if pil_image is not None else None
            outputs = pipe.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
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
        finally:
            pipe.release()
        decoded = processor.detokenize(sequences=outputs.sequences)
        sequences = decoded[0]
        return sequences[0] if num_return_sequences == 1 else sequences
