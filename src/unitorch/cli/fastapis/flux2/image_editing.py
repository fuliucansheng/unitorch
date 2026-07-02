# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
from typing import List, Optional, Union

import torch
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from diffusers.utils import numpy_to_pil
from torch import autocast

from unitorch.utils import is_bfloat16_available
from unitorch.cli import (
    Config,
    GenericFastAPI,
    config_defaults_init,
    config_defaults_method,
    register_fastapi,
)
from unitorch.cli.fastapis.flux2.text2image import Flux2FastAPIPipeline
from unitorch.cli import PipelineReplicaPool


class Flux2ImageEditingFastAPIPipeline(Flux2FastAPIPipeline):
    @classmethod
    @config_defaults_init("core/fastapi/pipeline/flux2/editing")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        added_tokens: Optional[str] = None,
        tokenizer_class: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        return cls._from_config_section(
            config,
            "core/fastapi/pipeline/flux2/editing",
            pretrained_name=pretrained_name,
            pretrained_weight_path=pretrained_weight_path,
            tokenizer_path=tokenizer_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            added_tokens=added_tokens,
            tokenizer_class=tokenizer_class,
            device=device,
            pretrained_lora_names=pretrained_lora_names,
            pretrained_lora_weights_path=pretrained_lora_weights_path,
            pretrained_lora_weights=pretrained_lora_weights,
            pretrained_lora_alphas=pretrained_lora_alphas,
            **kwargs,
        )

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    @config_defaults_method("core/fastapi/pipeline/flux2/editing")
    def __call__(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 4,
        seed: Optional[int] = 1123,
    ):
        if self._enable_cpu_offload:
            raise NotImplementedError(
                "FLUX.2 FastAPI CPU offload is not supported with explicit tokenizer generation."
            )

        inputs = self.processor.editing_inputs(
            prompt=text,
            refer_image=image,
            max_seq_length=self.max_sequence_length,
        )
        input_ids = inputs.input_ids.unsqueeze(0).to(self.device)
        attention_mask = inputs.attention_mask.unsqueeze(0).to(self.device)
        refer_pixel_values = inputs.refer_pixel_values.unsqueeze(0).to(self.device)
        prompt_embeds, text_ids = self._encode_prompt(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self.seed = seed
        outputs = self._generate_from_embeds(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            refer_pixel_values=refer_pixel_values,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_infer_timesteps=num_timesteps,
        )
        return numpy_to_pil(outputs.images.cpu().numpy())[0]


@register_fastapi("core/fastapi/flux2/editing")
class Flux2ImageEditingFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/flux2/editing")
        self._section = "core/fastapi/flux2/editing"
        router = config.getoption("router", "/core/fastapi/flux2/editing")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["POST"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(
        self,
        pretrained_name: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
    ):
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
            Flux2ImageEditingFastAPIPipeline.from_config(
                self.config,
                pretrained_name=pretrained_name,
                pretrained_lora_names=pretrained_lora_names,
                pretrained_lora_weights=pretrained_lora_weights,
                pretrained_lora_alphas=pretrained_lora_alphas,
            )
            for _ in range(num_replicas)
        ]
        for pipe, device in zip(pipelines, devices):
            if device is not None and hasattr(pipe, "to"):
                pipe.to(device)
        self._pipes = PipelineReplicaPool(pipelines, lock=lock)
        return "start success"

    def stop(self):
        if self._pipes is not None:
            self._pipes.close()
        self._pipes = None
        return "stop success"

    def status(self):
        return "running" if self._pipes is not None else "stopped"

    async def generate(
        self,
        text: str,
        image: UploadFile,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 4,
        seed: Optional[int] = 1123,
    ):
        assert self._pipes is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        pipe = self._pipes.acquire()
        try:
            image = pipe(
                text=text,
                image=image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )
        finally:
            pipe.release()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
