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
from unitorch.cli.pipelines.qwen_vl import (
    QWen2_5VLForGenerationPipeline,
)


@register_fastapi("core/fastapi/qwen2_5_vl")
class QWen2_5VLFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/qwen2_5_vl")
        router = config.getoption("router", "/core/fastapi/qwen2_5_vl")
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

    def start(self, pretrained_name: str = "qwen2_5-vl-3b-instruct"):
        self._pipe = QWen2_5VLForGenerationPipeline.from_core_configure(
            self.config,
            pretrained_name=pretrained_name,
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
        image: UploadFile = File(...),
        use_chat_template: Optional[bool] = True,
    ):
        assert self._pipe is not None
        image = await image.read()
        image = Image.open(io.BytesIO(image)).convert("RGB")
        async with self._lock:
            result = self._pipe(
                text,
                images=image,
                use_chat_template=use_chat_template,
                lora_checkpoints=[],
                lora_weights=[],
                lora_alphas=[],
                lora_urls=[],
                lora_files=[],
            )

        return result
