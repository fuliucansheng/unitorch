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
from torch import autocast
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil

from diffusers.pipelines import (
    WanImageToVideoPipeline,
)

from unitorch.utils import is_remote_url, tensor2vid
from unitorch.models.diffusers import WanForImage2VideoGeneration
from unitorch.models.diffusers import WanProcessor
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch.cli.pipelines import Schedulers
from unitorch.cli.models.diffusion_utils import export_to_video


class WanForImage2VideoFastAPIPipeline(WanForImage2VideoGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        config2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            config2_path=config2_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
        )
        self.processor = WanProcessor(
            vocab_path=vocab_path,
            vae_config_path=vae_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/wan/image2video")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        config2_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/wan/image2video")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "wan-v2.2-i2v-14b"
        )
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        config2_path = config2_path or config.getoption("config2_path", None)
        config2_path = pop_value(
            config2_path,
            nested_dict_value(pretrained_infos, "transformer2", "config"),
        )

        if config2_path is not None:
            config2_path = cached_path(config2_path)

        text_config_path = text_config_path or config.getoption(
            "text_config_path", None
        )
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = vae_config_path or config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = scheduler_config_path or config.getoption(
            "scheduler_config_path", None
        )
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        quant_config_path = quant_config_path or config.getoption(
            "quant_config_path", None
        )
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer2", "weight"),
                    prefix_keys={"": "transformer2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]

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
                pretrained_stable_extensions_infos,
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
                    pretrained_stable_extensions_infos, name, "lora", "weight"
                )
                for name in pretrained_lora_names
            ]
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_weights)
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_alphas)

        lora_weights_path = pretrained_lora_weights_path or config.getoption(
            "pretrained_lora_weights_path", None
        )

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            config2_path=config2_path,
            vocab_path=vocab_path,
            quant_config_path=quant_config_path,
            weight_path=weight_path,
            state_dict=state_dict,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
        )
        return inst

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    @add_default_section_for_function("core/fastapi/pipeline/wan/image2video")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        neg_text: Optional[str] = "",
        num_frames: Optional[int] = 81,
        num_fps: Optional[int] = 16,
        guidance_scale: Optional[float] = 5.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        image = image.convert("RGB")
        inputs = self.processor.image2video_inputs(
            text,
            image,
            negative_prompt=neg_text,
            max_seq_length=77,
        )
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs["input_ids"],
            negative_input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["attention_mask"],
            negative_attention_mask=inputs["negative_attention_mask"],
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        outputs = self.pipeline(
            image=inputs["vae_pixel_values"],
            prompt_embeds=prompt_outputs.prompt_embeds,
            negative_prompt_embeds=prompt_outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(seed),
            num_inference_steps=num_timesteps,
            height=inputs["vae_pixel_values"].size(-2),
            width=inputs["vae_pixel_values"].size(-1),
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        )

        frames = tensor2vid(outputs.frames.float())
        name = export_to_video(frames, fps=num_fps)
        return name


@register_fastapi("core/fastapi/wan/image2video")
class WanForImage2VideoFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/wan/image2video")
        router = config.getoption("router", "/core/fastapi/wan/image2video")
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["POST"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(
        self,
        pretrained_name: Optional[str] = "wan-v2.2-i2v-14b",
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
    ):
        self._pipe = WanForImage2VideoFastAPIPipeline.from_core_configure(
            self.config,
            pretrained_name=pretrained_name,
            pretrained_lora_names=pretrained_lora_names,
            pretrained_lora_weights=pretrained_lora_weights,
            pretrained_lora_alphas=pretrained_lora_alphas,
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
        neg_text: Optional[str] = "",
        num_frames: Optional[int] = 81,
        num_fps: Optional[int] = 16,
        guidance_scale: Optional[float] = 5.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            video = self._pipe(
                text,
                image,
                neg_text=neg_text,
                num_frames=num_frames,
                num_fps=num_fps,
                guidance_scale=guidance_scale,
                strength=strength,
                num_timesteps=num_timesteps,
                seed=seed,
            )
        buffer = io.BytesIO()
        with open(video, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=output.mp4"},
        )
