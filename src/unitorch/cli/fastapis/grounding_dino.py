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
from unitorch.models.grounding_dino import (
    GroundingDinoForDetection as _GroundingDinoForDetection,
)
from unitorch.models.grounding_dino import GroundingDinoProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    register_fastapi,
    cached_path,
    config_defaults_init,
    config_defaults_method,
    Config,
    GenericFastAPI,
)
from unitorch.cli.models.grounding_dino import pretrained_grounding_dino_infos


class GroundingDinoForDetectionPipeline(_GroundingDinoForDetection):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        vision_config_path: str,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = GroundingDinoProcessor(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/grounding_dino")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/grounding_dino")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "grounding-dino-tiny"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "config"
            ),
        )
        config_path = cached_path(config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "vocab"
            ),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "vision_config"
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
            nested_dict_value(
                pretrained_grounding_dino_infos, pretrained_name, "weight"
            ),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path,
            vision_config_path,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/grounding_dino")
    def __call__(
        self,
        text: str,
        image: Union[Image.Image, str],
        text_threshold: Optional[float] = 0.25,
        box_threshold: Optional[float] = 0.25,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.detection_inputs(text, image)
        outputs = self.detect(
            inputs.pixel_values.unsqueeze(0).to(self._device),
            inputs.input_ids.unsqueeze(0).to(self._device),
            inputs.token_type_ids.unsqueeze(0).to(self._device),
            inputs.attention_mask.unsqueeze(0).to(self._device),
            norm_bboxes=True,
            text_threshold=text_threshold,
            box_threshold=box_threshold,
        )
        result_image = image.copy()
        bboxes = outputs["bboxes"][0].cpu().numpy()
        scores = outputs["scores"][0].cpu().numpy()
        classes = outputs["classes"][0].cpu().numpy()
        for bbox, score, tokens in zip(bboxes, scores, classes):
            if score < box_threshold:
                continue
            classid = self.processor.tokenizer.decode(
                [t for t in tokens if t != 0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
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


@register_fastapi("core/fastapi/grounding_dino")
class GroundingDinoForDetectionFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/grounding_dino")
        router = config.getoption("router", "/core/fastapi/grounding_dino")
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

    def start(self, pretrained_name: Optional[str] = "grounding-dino-tiny"):
        self._pipe = GroundingDinoForDetectionPipeline.from_config(
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
        image: UploadFile,
        text_threshold: float = 0.25,
        box_threshold: float = 0.25,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            result_image = self._pipe(
                text,
                image,
                text_threshold=text_threshold,
                box_threshold=box_threshold,
            )

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
