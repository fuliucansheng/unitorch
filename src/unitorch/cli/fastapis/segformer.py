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
from unitorch.models.segformer import (
    SegformerForSegmentation as _SegformerForSegmentation,
)
from unitorch.models.segformer import SegformerProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    register_fastapi,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    CoreConfigureParser,
    GenericFastAPI,
)
from unitorch.cli.models.segformer import pretrained_segformer_infos


class SegformerForSegmentationPipeline(_SegformerForSegmentation):
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
        self.processor = SegformerProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/segformer")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/segformer")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "segformer-swin-tiny-ade-semantic"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_segformer_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_segformer_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_segformer_infos, pretrained_name, "weight"),
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
    @add_default_section_for_function("core/fastapi/pipeline/segformer")
    def __call__(
        self,
        image: Union[Image.Image, str],
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        width, height = image.size
        inputs = self.processor.classification(image)
        outputs = self.forward(inputs.pixel_values.unsqueeze(0).to(self._device))
        batch = outputs.logits.shape[0]
        num_classes = outputs.logits.shape[-1]
        masks = torch.softmax(outputs.logits, dim=1)
        masks = masks * (masks == masks.max(dim=1, keepdim=True).values).float()
        classes = (
            torch.arange(num_classes, device=masks.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        masks = torch.nn.functional.interpolate(
            masks,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks.cpu().numpy()[0]
        classes = classes.cpu().numpy()[0]
        id2label = self.segformer.config.id2label
        labels = [id2label.get(int(cls), None) for cls in classes]
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return [
            (mask, label) for mask, label in zip(masks, labels) if label is not None
        ]


@register_fastapi("core/fastapi/segformer")
class SegformerForSegmentationFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/fastapi/segformer")
        router = config.getoption("router", "/core/fastapi/segformer")
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

    def start(self, pretrained_name: Optional[str] = "segformer-swin-tiny-ade-semantic"):
        self._pipe = SegformerForSegmentationPipeline.from_core_configure(
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
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            results = self._pipe(image)

        return [(mask.tolist(), label) for mask, label in results]
