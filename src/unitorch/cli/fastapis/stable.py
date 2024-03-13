# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli import register_fastapi
from unitorch.cli.pipelines.stable import StableForText2ImageGenerationPipeline


@register_fastapi("core/fastapi/stable/text2image")
class StableText2ImageFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/stable/text2image")
        router = config.getoption("router", "core/fastapi/stable/text2image")
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/", self.serve, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self, **kwargs):
        self._pipe = StableForText2ImageGenerationPipeline.from_core_configure(
            self.config
        )

    def stop(self, **kwargs):
        del self._pipe
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe

    def serve(
        self,
        text: str,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image = self._pipe(
            text,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
