# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli import register_fastapi
from unitorch.cli.fastapis.stable.text2image import StableForText2ImageFastAPIPipeline
from unitorch.cli.fastapis.stable.image2video import StableForImage2VideoFastAPIPipeline


@register_fastapi("core/fastapi/stable/text2image")
class StableText2ImageFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/stable/text2image")
        router = config.getoption("router", "/core/fastapi/stable/text2image")
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/", self.serve, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self, **kwargs):
        self._pipe = StableForText2ImageFastAPIPipeline.from_core_configure(self.config)

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


@register_fastapi("core/fastapi/stable/image2video")
class StableImage2VideoFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/stable/image2video")
        router = config.getoption("router", "/core/fastapi/stable/image2video")
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/", self.serve, methods=["POST"])

    @property
    def router(self):
        return self._router

    def start(self, **kwargs):
        self._pipe = StableForImage2VideoFastAPIPipeline.from_core_configure(
            self.config
        )

    def stop(self, **kwargs):
        del self._pipe
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe

    async def serve(
        self,
        image: UploadFile,
        num_frames: Optional[int] = 30,
        num_fps: Optional[int] = 6,
        min_guidance_scale: Optional[float] = 1.0,
        max_guidance_scale: Optional[float] = 2.5,
        motion_bucket_id: Optional[int] = 127,
        decode_chunk_size: Optional[int] = 8,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        video = self._pipe(
            image,
            num_frames=num_frames,
            num_fps=num_fps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            motion_bucket_id=motion_bucket_id,
            decode_chunk_size=decode_chunk_size,
            num_timesteps=num_timesteps,
            seed=seed,
        )
        buffer = io.BytesIO()
        with open(video, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="video/mp4",
        )
