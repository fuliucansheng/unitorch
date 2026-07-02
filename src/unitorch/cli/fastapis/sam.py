# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
from unitorch.models.sam import (
    SamForSegmentation as _SamForSegmentation,
)
from unitorch.models.sam import SamProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.utils import is_remote_url
from unitorch.cli import (
    register_fastapi,
    cached_path,
    config_defaults_init,
    config_defaults_method,
    Config,
    GenericFastAPI,
)
from unitorch.cli.models.sam import (
    pretrained_sam_infos,
    pretrained_sam_extensions_infos,
)
from unitorch.cli import PipelineReplicaPool


class SamForSegmentationPipeline(_SamForSegmentation):
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
        self.processor = SamProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/sam")
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
        config.set_default_section("core/fastapi/pipeline/sam")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "sam-vit-base"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "weight"),
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
    @config_defaults_method("core/fastapi/pipeline/sam")
    def __call__(
        self,
        image: Union[Image.Image, str],
        points: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        mask_threshold: Optional[float] = 0.1,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.vision_processor(image)
        pixel_values, original_sizes, reshaped_input_sizes = (
            inputs["pixel_values"],
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        pixel_values = [torch.from_numpy(pixel_value) for pixel_value in pixel_values]
        pixel_values = torch.stack(pixel_values).to(self._device)
        input_points = (
            torch.Tensor([[points]]).to(self._device) if points is not None else None
        )
        input_boxes = (
            torch.Tensor([boxes]).to(self._device) if boxes is not None else None
        )
        if input_points is not None:
            input_points[:, :, :, 1] = (
                input_points[:, :, :, 1]
                / original_sizes[0][0]
                * reshaped_input_sizes[0][0]
            )
            input_points[:, :, :, 0] = (
                input_points[:, :, :, 0]
                / original_sizes[0][1]
                * reshaped_input_sizes[0][1]
            )
        if input_boxes is not None:
            input_boxes[:, :, :, 0] = (
                input_boxes[:, :, :, 0]
                / original_sizes[0][1]
                * reshaped_input_sizes[0][1]
            )
            input_boxes[:, :, :, 1] = (
                input_boxes[:, :, :, 1]
                / original_sizes[0][0]
                * reshaped_input_sizes[0][0]
            )
            input_boxes[:, :, :, 2] = (
                input_boxes[:, :, :, 2]
                / original_sizes[0][1]
                * reshaped_input_sizes[0][1]
            )
            input_boxes[:, :, :, 3] = (
                input_boxes[:, :, :, 3]
                / original_sizes[0][0]
                * reshaped_input_sizes[0][0]
            )
        outputs = self.segment(
            pixel_values,
            input_points=input_points,
            input_boxes=input_boxes,
        )
        processed_masks = self.processor.vision_processor.post_process_masks(
            outputs.masks,
            original_sizes,
            reshaped_input_sizes,
            mask_threshold=mask_threshold,
            binarize=True,
        )[0]
        if len(processed_masks) == 0:
            return None
        first_mask = processed_masks[0, 0].permute(0, 1)
        first_mask = first_mask.cpu().to(torch.uint8) * 255
        mask_image = Image.fromarray(np.array(first_mask))
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return mask_image


@register_fastapi("core/fastapi/sam")
class SamForSegmentationFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/sam")
        self._section = "core/fastapi/sam"
        router = config.getoption("router", "/core/fastapi/sam")
        self._pipes = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])

    @property
    def router(self):
        return self._router

    def start(self, pretrained_name: Optional[str] = "sam-vit-base"):
        num_replicas = int(
            self.config.getdefault(self._section, "pipeline_num_replicas", 1)
        )
        devices = self.config.getdefault(
            self._section, "pipeline_replica_devices", "cpu"
        )
        lock = self.config.getdefault(
            self._section, "pipeline_replica_lock", True
        )
        if devices is None:
            devices = []
        if isinstance(devices, str):
            devices = [devices] * num_replicas
        pipelines = [
            SamForSegmentationPipeline.from_config(
                self.config,
                pretrained_name=pretrained_name,
            )
            for _ in range(num_replicas)
        ]
        for pipe, device in zip(pipelines, devices):
            if device is not None and hasattr(pipe, "to"):
                pipe.to(device)
        self._pipes = PipelineReplicaPool(pipelines, lock=lock)
        return "start success"

    def stop(self):
        if self._pipes is not None:
            self._pipes.close()
        self._pipes = None
        return "stop success"

    def status(self):
        return "running" if self._pipes is not None else "stopped"

    async def generate(
        self,
        image: UploadFile,
        points: Optional[List] = None,
        boxes: Optional[List] = None,
        mask_threshold: float = 0.1,
    ):
        assert self._pipes is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        pipe = self._pipes.acquire()
        try:
            mask_image = pipe(
                image,
                points=points,
                boxes=boxes,
                mask_threshold=mask_threshold,
                
            )
        finally:
            pipe.release()
        if mask_image is None:
            return StreamingResponse(
                io.BytesIO(),
                media_type="image/png",
            )

        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
