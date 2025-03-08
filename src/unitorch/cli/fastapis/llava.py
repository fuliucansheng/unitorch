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
from unitorch.cli.pipelines.llava import (
    LlavaMistralClipForGenerationPipeline,
    LlavaLlamaSiglipForGenerationPipeline,
)


@register_fastapi("core/fastapi/llava/mistral_clip")
class LlavaMistralClipFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/llava/mistral_clip")
        router = config.getoption("router", "/core/fastapi/llava/mistral_clip")
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
        self._pipe = LlavaMistralClipForGenerationPipeline.from_core_configure(
            self.config,
            pretrained_name="llava-v1.6-mistral-7b-hf",
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
        text: str,
        image: UploadFile,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        text = f"[INST] <image>\n {text} [/INST]"
        async with self._lock:
            caption = self._pipe(
                text,
                image,
                lora_checkpoints=[],
                lora_weights=[],
                lora_alphas=[],
                lora_urls=[],
                lora_files=[],
            )

        return caption


@register_fastapi("core/fastapi/llava/joycaption2")
class LlavaLlamaSiglipFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/llava/joycaption2")
        router = config.getoption("router", "/core/fastapi/llava/joycaption2")
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
        self._pipe = LlavaLlamaSiglipForGenerationPipeline.from_core_configure(
            self.config,
            pretrained_name="llava-v1.6-joycaption-2",
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
        text: str,
        image: UploadFile,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        text = f"<|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\nToday Date: 26 July 2024\\n\\nYou are a helpful image captioner.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>{text}|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
        async with self._lock:
            caption = self._pipe(
                text,
                image,
                lora_checkpoints=[],
                lora_weights=[],
                lora_alphas=[],
                lora_urls=[],
                lora_files=[],
            )

        return caption
