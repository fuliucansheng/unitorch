# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import json
import logging
import torch
import pandas as pd
from PIL import Image
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from diffusers.models import ControlNetModel
from diffusers.pipelines import (
    WanImageToVideoPipeline,
)
from unitorch import is_xformers_available
from unitorch.utils import (
    is_remote_url,
    is_bfloat16_available,
)
from unitorch.models.diffusers import WanForImage2VideoGeneration
from unitorch.models.diffusers import WanProcessor

from unitorch.utils import pop_value, nested_dict_value, tensor2vid
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)

from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch.cli.pipelines import Schedulers
from unitorch.cli.models.diffusion_utils import export_to_video


class WanForImage2VideoGenerationPipeline(WanForImage2VideoGeneration):
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
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
        boundary_ratio: Optional[float] = 0.9,
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
            boundary_ratio=boundary_ratio,
        )
        self.processor = WanProcessor(
            vocab_path=vocab_path,
            vae_config_path=vae_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        self._enable_cpu_offload = enable_cpu_offload
        self._enable_xformers = enable_xformers

        if not self._enable_cpu_offload:
            self.image.to(device=self._device)

    @classmethod
    @add_default_section_for_init("core/pipeline/wan/image2video")
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
        **kwargs,
    ):
        config.set_default_section("core/pipeline/wan/image2video")
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
        enable_xformers = config.getoption("enable_xformers", False)

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
        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            scheduler_config_path=scheduler_config_path,
            vae_config_path=vae_config_path,
            config2_path=config2_path,
            vocab_path=vocab_path,
            quant_config_path=quant_config_path,
            weight_path=weight_path,
            state_dict=state_dict,
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
    @add_default_section_for_function("core/pipeline/wan/image2video")
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        num_fps: Optional[int] = 16,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        guidance_scale: Optional[float] = 5.0,
        lora_checkpoints: Optional[Union[str, List[str]]] = [],
        lora_weights: Optional[Union[float, List[float]]] = [],
        lora_alphas: Optional[Union[float, List[float]]] = [],
        lora_urls: Optional[Union[str, List[str]]] = [],
        lora_files: Optional[Union[str, List[str]]] = [],
    ):
        image = image.convert("RGB")
        inputs = self.processor.image2video_inputs(
            prompt,
            image.resize((width, height), resample=Image.LANCZOS),
            negative_prompt=negative_prompt,
            max_seq_length=77,
        )

        assert scheduler is None or scheduler in Schedulers
        if scheduler is not None:
            self.scheduler = Schedulers.get(scheduler).from_config(
                self.scheduler.config
            )

        self.pipeline = WanImageToVideoPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            transformer_2=getattr(self, "transformer2", None),
            scheduler=self.scheduler,
            tokenizer=None,
            image_processor=None,
            boundary_ratio=self.boundary_ratio,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
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
                processed_lora_files.append(
                    nested_dict_value(
                        pretrained_stable_extensions_infos, ckpt, "weight"
                    )
                )
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

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

        if self._enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

        outputs = self.get_prompt_outputs(
            input_ids=inputs["input_ids"],
            negative_input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["attention_mask"],
            negative_attention_mask=inputs["negative_attention_mask"],
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        outputs = self.pipeline(
            image=inputs["vae_pixel_values"],
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(seed),
            num_inference_steps=num_timesteps,
            height=inputs["vae_pixel_values"].size(-2),
            width=inputs["vae_pixel_values"].size(-1),
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        )

        self.unload_lora_weights()

        frames = tensor2vid(outputs.frames.float())
        name = export_to_video(frames, fps=num_fps)
        return name
