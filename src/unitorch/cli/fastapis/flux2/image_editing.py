# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import gc
import io
import asyncio
from typing import List, Optional, Union

import torch
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
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


class Flux2ImageEditingFastAPIPipeline(Flux2FastAPIPipeline):
    @classmethod
    @config_defaults_init("core/fastapi/pipeline/flux2/editing")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        processor_name_or_path: Optional[str] = None,
        processor_subfolder: Optional[str] = None,
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
            processor_name_or_path=processor_name_or_path,
            processor_subfolder=processor_subfolder,
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
        guidance_scale: Optional[float] = 4.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        caption_upsample_temperature: Optional[float] = None,
    ):
        return super().__call__(
            text=text,
            image=image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
            caption_upsample_temperature=caption_upsample_temperature,
        )


@register_fastapi("core/fastapi/flux2/editing")
class Flux2ImageEditingFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/flux2/editing")
        router = config.getoption("router", "/core/fastapi/flux2/editing")
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["POST"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(
        self,
        pretrained_name: Optional[str] = "flux2-dev",
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
    ):
        self._pipe = Flux2ImageEditingFastAPIPipeline.from_config(
            self.config,
            pretrained_name=pretrained_name,
            pretrained_lora_names=pretrained_lora_names,
            pretrained_lora_weights=pretrained_lora_weights,
            pretrained_lora_alphas=pretrained_lora_alphas,
        )
        return "start success"

    def stop(self):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        return "stop success"

    def status(self):
        return "running" if self._pipe is not None else "stopped"

    async def generate(
        self,
        text: str,
        image: UploadFile,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 4.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        caption_upsample_temperature: Optional[float] = None,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        async with self._lock:
            image = self._pipe(
                text=text,
                image=image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
                caption_upsample_temperature=caption_upsample_temperature,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
