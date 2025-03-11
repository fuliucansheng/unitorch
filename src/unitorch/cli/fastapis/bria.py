# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import re
import gc
import json
import logging
import torch
import hashlib
import asyncio
import pandas as pd
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli.pipelines.bria import BRIAForSegmentationPipeline


@register_fastapi("core/fastapi/bria")
class BRIAFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/bria")
        router = config.getoption("router", "/core/fastapi/bria")
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.serve, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        self._pipe = BRIAForSegmentationPipeline.from_core_configure(
            self.config,
            pretrained_weight_path="https://huggingface.co/briaai/RMBG-2.0/resolve/main/pytorch_model.bin",
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

    async def serve(
        self,
        image: UploadFile,
        threshold: float = 0.5,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            mask = self._pipe(image, threshold=threshold)

        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
