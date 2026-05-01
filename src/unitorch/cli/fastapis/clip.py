# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import gc
import torch
import asyncio
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, UploadFile
from unitorch.utils import is_remote_url
from unitorch.models.clip import (
    ClipForClassification as _ClipForClassification,
    ClipForTextClassification as _ClipForTextClassification,
    ClipForImageClassification as _ClipForImageClassification,
    ClipForMatching as _ClipForMatching,
)
from unitorch.models.clip import ClipProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_fastapi,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli.models.clip import (
    pretrained_clip_infos,
    pretrained_clip_extensions_infos,
)


class ClipForClassificationPipeline(_ClipForClassification):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        max_seq_length: Optional[int] = 512,
        id2label: Optional[Dict[int, str]] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
        )
        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self.id2label = id2label
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/clip")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/clip")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "clip-vit-base-patch16"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = merge_path or config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        max_seq_length = config.getoption("max_seq_length", 512)
        id2label = id2label or config.getoption("id2label", None)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            max_seq_length=max_seq_length,
            id2label=id2label,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/fastapi/pipeline/clip")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 512,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        outputs = super().forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            pixel_values=inputs["pixel_values"],
        )
        scores = outputs.softmax(dim=-1).squeeze(0)
        if self.id2label is not None:
            return self.id2label[scores.argmax(-1).item()], scores.max(-1)[0].item()
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return scores.argmax(-1).item(), scores.max(-1)[0].item()


class ClipForTextClassificationPipeline(_ClipForTextClassification):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        max_seq_length: Optional[int] = 512,
        id2label: Optional[Dict[int, str]] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
        )
        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            max_seq_length=max_seq_length,
        )
        self.id2label = id2label
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/clip/text")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/clip/text")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "clip-vit-base-patch16"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = merge_path or config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        max_seq_length = config.getoption("max_seq_length", 512)
        id2label = id2label or config.getoption("id2label", None)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            max_seq_length=max_seq_length,
            id2label=id2label,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/fastapi/pipeline/clip/text")
    def __call__(
        self,
        text: str,
        max_seq_length: Optional[int] = 512,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.text_classification(
            text=text,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        outputs = super().forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
        )
        scores = outputs.softmax(dim=-1).squeeze(0)
        if self.id2label is not None:
            return self.id2label[scores.argmax(-1).item()], scores.max(-1)[0].item()
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return scores.argmax(-1).item(), scores.max(-1)[0].item()


class ClipForImageClassificationPipeline(_ClipForImageClassification):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        id2label: Optional[Dict[int, str]] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
        )
        self.processor = ClipProcessor(
            vision_config_path=vision_config_path,
        )
        self.id2label = id2label
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/clip/image")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/clip/image")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "clip-vit-base-patch16"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        max_seq_length = config.getoption("max_seq_length", 512)
        id2label = id2label or config.getoption("id2label", None)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path=vision_config_path,
            projection_dim=projection_dim,
            num_classes=num_classes,
            max_seq_length=max_seq_length,
            id2label=id2label,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/fastapi/pipeline/clip/image")
    def __call__(
        self,
        image: Image.Image,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.image_classification(
            image=image,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        outputs = super().forward(
            pixel_values=inputs["pixel_values"],
        )
        scores = outputs.softmax(dim=-1).squeeze(0)
        if self.id2label is not None:
            return self.id2label[scores.argmax(-1).item()], scores.max(-1)[0].item()
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return scores.argmax(-1).item(), scores.max(-1)[0].item()


class ClipForMatchingPipeline(_ClipForMatching):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        projection_dim: Optional[int] = 512,
        max_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
        )
        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/matching/clip")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/matching/clip")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "clip-vit-base-patch16"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = merge_path or config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        projection_dim = config.getoption("projection_dim", 512)
        max_seq_length = config.getoption("max_seq_length", 512)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            projection_dim=projection_dim,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/fastapi/pipeline/matching/clip")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 77,
        lora_checkpoints: Optional[Union[str, List[str]]] = [],
        lora_weights: Optional[Union[float, List[float]]] = [],
        lora_alphas: Optional[Union[float, List[float]]] = [],
        lora_urls: Optional[Union[str, List[str]]] = [],
        lora_files: Optional[Union[str, List[str]]] = [],
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        if isinstance(lora_checkpoints, str):
            lora_checkpoints = [lora_checkpoints]
        if isinstance(lora_weights, float):
            lora_weights = [lora_weights]
        if isinstance(lora_alphas, float):
            lora_alphas = [lora_alphas]
        if isinstance(lora_urls, str):
            lora_urls = [lora_urls]
        if isinstance(lora_files, str):
            lora_files = [lora_files]

        assert (
            len(lora_checkpoints) == len(lora_weights)
            and len(lora_checkpoints) == len(lora_alphas)
            and len(lora_checkpoints) == len(lora_urls)
            and len(lora_checkpoints) == len(lora_files)
        )
        processed_lora_files, processed_lora_weights, processed_lora_alphas = [], [], []
        for ckpt, url, file, weight, alpha in zip(
            lora_checkpoints, lora_urls, lora_files, lora_weights, lora_alphas
        ):
            if ckpt is not None:
                lora_file = nested_dict_value(
                    pretrained_clip_extensions_infos, ckpt, "weight"
                )
                processed_lora_files.append(lora_file)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)
            elif url is not None and is_remote_url(url):
                processed_lora_files.append(url)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)
            elif file is not None:
                processed_lora_files.append(file)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)

        if len(processed_lora_files) > 0:
            self.load_lora_weights(
                processed_lora_files,
                lora_weights=processed_lora_weights,
                lora_alphas=processed_lora_alphas,
            )

        outputs = super().forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            pixel_values=inputs["pixel_values"],
        )
        scores = outputs.sigmoid().squeeze(0)
        self.unload_lora_weights()
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return scores[0].item()


@register_fastapi("core/fastapi/clip")
class ClipForClassificationFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/fastapi/clip")
        router = config.getoption("router", "/core/fastapi/clip")
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

    def start(self, pretrained_name: str = "clip-vit-base-patch16"):
        self._pipe = ClipForClassificationPipeline.from_core_configure(
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
        max_seq_length: Optional[int] = 512,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            result = self._pipe(
                text,
                image,
                max_seq_length=max_seq_length,
            )

        return result


@register_fastapi("core/fastapi/clip/text")
class ClipForTextClassificationFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/fastapi/clip/text")
        router = config.getoption("router", "/core/fastapi/clip/text")
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

    def start(self, pretrained_name: str = "clip-vit-base-patch16"):
        self._pipe = ClipForTextClassificationPipeline.from_core_configure(
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
        max_seq_length: Optional[int] = 512,
    ):
        assert self._pipe is not None
        async with self._lock:
            result = self._pipe(
                text,
                max_seq_length=max_seq_length,
            )

        return result


@register_fastapi("core/fastapi/clip/image")
class ClipForImageClassificationFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/fastapi/clip/image")
        router = config.getoption("router", "/core/fastapi/clip/image")
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

    def start(self, pretrained_name: str = "clip-vit-base-patch16"):
        self._pipe = ClipForImageClassificationPipeline.from_core_configure(
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
            result = self._pipe(image)

        return result


@register_fastapi("core/fastapi/clip/matching")
class ClipForMatchingFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section("core/fastapi/clip/matching")
        router = config.getoption("router", "/core/fastapi/clip/matching")
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

    def start(self, pretrained_name: str = "clip-vit-base-patch16"):
        self._pipe = ClipForMatchingPipeline.from_core_configure(
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
        max_seq_length: Optional[int] = 77,
        lora_checkpoints: Optional[Union[str, List[str]]] = [],
        lora_weights: Optional[Union[float, List[float]]] = [],
        lora_alphas: Optional[Union[float, List[float]]] = [],
        lora_urls: Optional[Union[str, List[str]]] = [],
        lora_files: Optional[Union[str, List[str]]] = [],
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            result = self._pipe(
                text,
                image,
                max_seq_length=max_seq_length,
                lora_checkpoints=lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                lora_urls=lora_urls,
                lora_files=lora_files,
            )

        return result
