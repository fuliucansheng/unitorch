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
    QwenImagePipeline,
    QwenImageEditPipeline,
)
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.models.diffusers import GenericQWenImageModel
from unitorch.models.diffusers import QWenImageProcessor

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


class QWenImageForImageEditingFastAPIPipeline(GenericQWenImageModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 12800,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = QWenImageProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            vision_config_path=vision_config_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)

        self.eval()

        self.pipeline = QwenImageEditPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            processor=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload
        self._enable_xformers = enable_xformers

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

        if self._enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

    @classmethod
    @add_default_section_for_init("core/fastapi/pipeline/qwen_image/editing")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/qwen_image/editing")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "qwen-image-editing"
        )
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

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

        merge_path = merge_path or config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        tokenizer_config = tokenizer_config or config.getoption(
            "tokenizer_config", None
        )
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(pretrained_infos, "text", "tokenizer_config"),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config) if tokenizer_config is not None else None
        )

        special_tokens_map = special_tokens_map or config.getoption(
            "special_tokens_map", None
        )
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(pretrained_infos, "text", "special_tokens_map"),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_infos, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        quant_config_path = quant_config_path or config.getoption(
            "quant_config_path", None
        )
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 12800)
        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", False)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={
                        "^visual.": "text.model.",
                        "^model(?!\.model).": "text.model.language_",
                        "^lm_head.": "text.",
                    },
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
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            quant_config_path=quant_config_path,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            state_dict=state_dict,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    @add_default_section_for_function("core/fastapi/pipeline/qwen_image/editing")
    def __call__(
        self,
        text: str,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        neg_text: Optional[str] = "",
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 4.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        inputs = self.processor.editing_inputs(
            text,
            refer_image=image,
            negative_prompt=neg_text,
        )
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["refer_pixel_values"],
            image_grid_thw=inputs["refer_image_grid_thw"],
            negative_input_ids=inputs.get("negative_input_ids", None),
            negative_attention_mask=inputs.get("negative_attention_mask", None),
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
            prompt_start_index=64,
        )
        prompt_embeds = prompt_outputs.prompt_embeds
        prompt_embeds_mask = prompt_outputs.prompt_embeds_mask
        negative_prompt_embeds = prompt_outputs.negative_prompt_embeds
        negative_prompt_embeds_mask = prompt_outputs.negative_prompt_embeds_mask

        outputs = self.pipeline(
            image=inputs["refer_vae_pixel_values"],
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            height=height,
            width=width,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=num_timesteps,
            guidance_scale=1.0,
            true_cfg_scale=guidance_scale,
            output_type="np.array",
        )

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0].resize((width, height))


@register_fastapi("core/fastapi/qwen_image/editing")
class QWenImageEditingFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"core/fastapi/qwen_image/editing")
        router = config.getoption("router", "/core/fastapi/qwen_image/editing")
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
        pretrained_name: Optional[str] = "qwen-image-editing",
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
    ):
        self._pipe = QWenImageForImageEditingFastAPIPipeline.from_core_configure(
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
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 2.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            image = self._pipe(
                text,
                image=image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
