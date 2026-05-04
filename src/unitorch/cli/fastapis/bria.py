# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import gc
import torch
import asyncio
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bria import (
    BRIAProcessor,
    BRIAForSegmentation as _BRIAForSegmentation,
)
from unitorch.cli import cached_path, config_defaults_init, config_defaults_method
from unitorch.cli import register_fastapi
from unitorch.cli import Config, GenericFastAPI


class BRIAForSegmentationPipeline(_BRIAForSegmentation):
    def __init__(
        self,
        image_size: Optional[int] = 1024,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__()
        self.processor = BRIAProcessor(image_size=image_size)
        self._device = "cpu" if device == "cpu" else int(device)
        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/bria")
    def from_config(cls, config, pretrained_weight_path=None, device=None, **kwargs):
        config.set_default_section("core/fastapi/pipeline/bria")
        image_size = config.getoption("image_size", 1024)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        weight_path = pretrained_weight_path or config.getoption("pretrained_weight_path", None)
        return cls(image_size=image_size, weight_path=weight_path, enable_cpu_offload=enable_cpu_offload, device=device)

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/bria")
    def __call__(self, image, threshold=0.5):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.segmentation_inputs(image)
        pixel_values = inputs.image.unsqueeze(0).to(self._device)
        outputs = self.forward(pixel_values).logits
        masks = [(mask.squeeze(0).cpu().numpy() > threshold).astype(np.uint8) for mask in outputs][0]
        result_image = Image.fromarray(masks * 255)
        result_image = result_image.resize(image.size, resample=Image.LANCZOS)
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return result_image


@register_fastapi("core/fastapi/bria")
class BRIAFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section(f"core/fastapi/bria")
        router = config.getoption("router", "/core/fastapi/bria")
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

    def start(self):
        self._pipe = BRIAForSegmentationPipeline.from_config(
            self.config,
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/hubfiles/resolve/main/bria_rmbg2.0_pytorch_model.bin",
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
