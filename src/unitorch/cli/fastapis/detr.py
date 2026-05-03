# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import gc
import torch
import asyncio
import numpy as np
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from unitorch.models.detr import (
    DetrForDetection as _DetrForDetection,
)
from unitorch.models.detr import DetrProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    register_fastapi,
    cached_path,
    config_defaults_init,
    config_defaults_method,
    Config,
    GenericFastAPI,
)
from unitorch.cli.models.detr import pretrained_detr_infos


class DetrForDetectionPipeline(_DetrForDetection):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = DetrProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/detr")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/detr")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "detr-resnet-50"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_detr_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/detr")
    def __call__(
        self,
        image: Union[Image.Image, str],
        threshold: Optional[float] = 0.5,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.image(image)
        pixel_values = [inputs.image.to(self._device)]
        outputs = self.detect(
            pixel_values,
            norm_bboxes=True,
            threshold=threshold,
        )
        result_image = image.copy()
        bboxes = outputs["bboxes"][0].cpu().numpy()
        scores = outputs["scores"][0].cpu().numpy()
        classes = outputs["classes"][0].cpu().numpy()
        for bbox, score, classid in zip(bboxes, scores, classes):
            if score < threshold:
                continue
            bbox = bbox * np.array(
                [
                    result_image.width,
                    result_image.height,
                    result_image.width,
                    result_image.height,
                ]
            )
            bbox = list(map(int, bbox))
            result_image = result_image.copy()
            draw = ImageDraw.Draw(result_image)
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0] + 5, bbox[1] + 5), f"{classid}", fill="blue")
            draw.text((bbox[0] + 5, bbox[1] + 15), f"{score:.2f}", fill="green")

        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return result_image


@register_fastapi("core/fastapi/detr")
class DetrForDetectionFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/detr")
        router = config.getoption("router", "/core/fastapi/detr")
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

    def start(self, pretrained_name: Optional[str] = "detr-resnet-50"):
        self._pipe = DetrForDetectionPipeline.from_config(
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
        image: UploadFile,
        threshold: float = 0.5,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            result_image = self._pipe(image, threshold=threshold)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
