# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli import register_fastapi
from unitorch.cli.pipelines.stable import StableForText2ImageGenerationPipeline


@register_fastapi("core/fastapi/stable/text2image")
class StableText2ImageFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self._model = None
        self._router = APIRouter(prefix="/core/fastapi/stable/text2image")
        self._router.add_api_route("/", self.serve, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self, **kwargs):
        self._model = StableForText2ImageGenerationPipeline.from_core_configure(
            self.config
        )

    def stop(self, **kwargs):
        del self._model
        self._model = None

    def serve(
        self,
        text: str,
        height: int = 512,
        width: int = 512,
    ):
        assert self._model is not None
        image = self._model(text, height, width)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
