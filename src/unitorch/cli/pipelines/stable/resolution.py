# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import json
import logging
import torch
import hashlib
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from diffusers.models import ControlNetModel
from diffusers.pipelines import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionDepth2ImgPipeline,
    StableVideoDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
)
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.models.diffusers import GenericStableModel
from unitorch.models.diffusers import StableProcessor

from unitorch.utils import pop_value, nested_dict_value
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


class StableForImageResolutionPipeline(GenericStableModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        pad_token: Optional[str] = "<|endoftext|>",
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
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
        self.processor = StableProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            pad_token=pad_token,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        self._enable_cpu_offload = enable_cpu_offload
        self._enable_xformers = enable_xformers

    @classmethod
    @add_default_section_for_init("core/pipeline/stable/resolution")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "stable-v1.5",
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/stable/resolution")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", text_config_path)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", vae_config_path)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption(
            "scheduler_config_path", scheduler_config_path
        )
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = config.getoption("vocab_path", vocab_path)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", merge_path)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        quant_config_path = config.getoption("quant_config_path", quant_config_path)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        pad_token = config.getoption("pad_token", "<|endoftext|>")
        weight_path = config.getoption("pretrained_weight_path", pretrained_weight_path)
        device = config.getoption("device", device)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(nested_dict_value(pretrained_infos, "unet", "weight")),
                load_weight(nested_dict_value(pretrained_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrained_infos, "vae", "weight")),
            ]

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            quant_config_path=quant_config_path,
            pad_token=pad_token,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            state_dict=state_dict,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/stable/resolution")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        neg_text: Optional[str] = "",
        guidance_scale: Optional[float] = 7.5,
        noise_level: Optional[int] = 20,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_params: Optional[Tuple[float, float, float, float]] = (
            0.9,
            0.2,
            1.2,
            1.4,
        ),
        lora_checkpoints: Optional[Union[str, List[str]]] = [],
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        lora_urls: Optional[Union[str, List[str]]] = [],
        lora_files: Optional[Union[str, List[str]]] = [],
    ):
        text_inputs = self.processor.text2image_inputs(
            text,
            negative_prompt=neg_text,
        )
        image_inputs = self.processor.image2image_inputs(image)

        assert scheduler is None or scheduler in Schedulers
        if scheduler is not None:
            self.scheduler = Schedulers.get(scheduler).from_config(
                self.scheduler.config
            )

        self.pipeline = StableDiffusionUpscalePipeline(
            vae=self.vae,
            text_encoder=self.text,
            unet=self.unet,
            scheduler=self.scheduler,
            low_res_scheduler=self.scheduler,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
        )
        inputs = {**text_inputs, **image_inputs}
        self.pipeline.set_progress_bar_config(disable=True)
        # if freeu_params is not None:
        #     self.pipeline.enable_freeu(*freeu_params)
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

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs.get("input_ids"),
            negative_input_ids=inputs.get("negative_input_ids"),
            attention_mask=inputs.get("attention_mask"),
            negative_attention_mask=inputs.get("negative_attention_mask"),
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        prompt_embeds = prompt_outputs.prompt_embeds
        negative_prompt_embeds = prompt_outputs.negative_prompt_embeds

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

        if self._enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

        outputs = self.pipeline(
            image=inputs["pixel_values"],
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=num_timesteps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            output_type="np.array",
        )
        self.unload_lora_weights()

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0]
