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
    ClipForImageClassificationV2 as _ClipForImageClassificationV2,
    ClipForMatching as _ClipForMatching,
)
from unitorch.models.clip import ClipProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_fastapi,
)
from unitorch.cli import Config, GenericFastAPI
from unitorch.cli.models import ACT2FN
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
    @config_defaults_init("core/fastapi/pipeline/clip")
    def from_config(
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
            id2label=id2label,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/clip")
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
    @config_defaults_init("core/fastapi/pipeline/clip/text")
    def from_config(
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
    @config_defaults_method("core/fastapi/pipeline/clip/text")
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
    @config_defaults_init("core/fastapi/pipeline/clip/image")
    def from_config(
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
    @config_defaults_method("core/fastapi/pipeline/clip/image")
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

class ClipForImageClassificationV2Pipeline(_ClipForImageClassificationV2):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        label_dict: Dict[str, str],
        projection_dim: Optional[int] = None,
        max_seq_length: Optional[int] = 128,
        output_embed_dim: Optional[int] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
        act_fn: Optional[str] = None,
        freeze_base_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
    ):
        if label_dict is None or len(label_dict) == 0:
            raise ValueError("label_dict must be provided")

        self.label_keys = list(label_dict.keys())
        self.label_values = list(label_dict.values())
        self.act_fn = ACT2FN.get(act_fn, None)
        super().__init__(
            config_path=config_path,
            labels=self.label_values,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            projection_dim=projection_dim,
            output_embed_dim=output_embed_dim,
            max_seq_length=max_seq_length,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/clip/image/v2")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        label_dict: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        act_fn: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/clip/image/v2")
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

        projection_dim = config.getoption("projection_dim", None)
        max_seq_length = config.getoption("max_seq_length", 128)
        output_embed_dim = config.getoption("output_embed_dim", None)
        freeze_base_model = config.getoption("freeze_base_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
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

        label_dict = (
            config.getoption("label_dict", None) if label_dict is None else label_dict
        )
        act_fn = config.getoption("act_fn", None) if act_fn is None else act_fn
        pretrained_lora_names = pretrained_lora_names or config.getoption(
            "pretrained_lora_names", None
        )
        pretrained_lora_weights = pretrained_lora_weights or config.getoption(
            "pretrained_lora_weights", 1.0
        )
        pretrained_lora_alphas = pretrained_lora_alphas or config.getoption(
            "pretrained_lora_alphas", 32.0
        )

        if (
            isinstance(pretrained_lora_names, str)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_clip_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif (
            isinstance(pretrained_lora_names, list)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_clip_extensions_infos,
                    name,
                    "lora",
                    "weight",
                )
                for name in pretrained_lora_names
            ]

        lora_weights_path = pretrained_lora_weights_path or config.getoption(
            "pretrained_lora_weights_path", None
        )

        return cls(
            config_path=config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            label_dict=label_dict,
            projection_dim=projection_dim,
            max_seq_length=max_seq_length,
            output_embed_dim=output_embed_dim,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
            act_fn=act_fn,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
        )

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/clip/image/v2")
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
        scores = outputs.squeeze(0).float().cpu().numpy()
        if self.act_fn is not None:
            scores = self.act_fn(scores)

        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()

        return {k: float(v) for k, v in zip(self.label_keys, scores)}


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
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
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
        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("core/fastapi/pipeline/matching/clip")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
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
        pretrained_lora_names = pretrained_lora_names or config.getoption(
            "pretrained_lora_names", None
        )
        pretrained_lora_weights = pretrained_lora_weights or config.getoption(
            "pretrained_lora_weights", 1.0
        )
        pretrained_lora_alphas = pretrained_lora_alphas or config.getoption(
            "pretrained_lora_alphas", 32.0
        )

        if (
            isinstance(pretrained_lora_names, str)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_clip_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif (
            isinstance(pretrained_lora_names, list)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_clip_extensions_infos,
                    name,
                    "lora",
                    "weight",
                )
                for name in pretrained_lora_names
            ]

        lora_weights_path = pretrained_lora_weights_path or config.getoption(
            "pretrained_lora_weights_path", None
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
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("core/fastapi/pipeline/matching/clip")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 77,
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
        scores = outputs.sigmoid().squeeze(0)
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return scores[0].item()


@register_fastapi("core/fastapi/clip")
class ClipForClassificationFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
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
        self._pipe = ClipForClassificationPipeline.from_config(
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
    def __init__(self, config: Config):
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
        self._pipe = ClipForTextClassificationPipeline.from_config(
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
    def __init__(self, config: Config):
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
        self._pipe = ClipForImageClassificationPipeline.from_config(
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

@register_fastapi("core/fastapi/clip/image/v2")
class ClipForImageClassificationV2FastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("core/fastapi/clip/image/v2")
        router = config.getoption("router", "/core/fastapi/clip/image/v2")
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

    def start(self, pretrained_name: Optional[str] = None):
        self._pipe = ClipForImageClassificationV2Pipeline.from_config(
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
    def __init__(self, config: Config):
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
        self._pipe = ClipForMatchingPipeline.from_config(
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
