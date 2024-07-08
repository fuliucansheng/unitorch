# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import json
import logging
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from unitorch import is_xformers_available
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
from unitorch.models.diffusers import StableForImage2VideoGeneration
from unitorch.models.diffusers import StableVideoProcessor
from unitorch.models.diffusers.modeling_stable import StableVideoDiffusionPipelineV2

from unitorch.utils import pop_value, nested_dict_value, tensor2vid
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch.cli.pipelines import Schedulers
from unitorch.cli.models.diffusion_utils import export_to_video


class StableForImage2VideoGenerationPipeline(StableForImage2VideoGeneration):
    def __init__(
        self,
        config_path: str,
        image_config_path: str,
        image_process_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            image_config_path=image_config_path,
            image_process_config_path=image_process_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = StableVideoProcessor(
            vision_config_path=image_process_config_path,
            vae_config_path=vae_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()
        self.to(device=self._device)

        self._enable_cpu_offload = enable_cpu_offload
        self._enable_xformers = enable_xformers

    @classmethod
    @add_default_section_for_init("core/pipeline/stable/image2video")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "stable-v1.5",
        config_path: Optional[str] = None,
        image_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        image_process_config_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("core/pipeline/stable/image2video")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)
        pretrain_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrain_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        image_config_path = config.getoption("image_config_path", image_config_path)
        image_config_path = pop_value(
            image_config_path,
            nested_dict_value(pretrain_infos, "image", "config"),
        )
        image_config_path = cached_path(image_config_path)

        vae_config_path = config.getoption("vae_config_path", vae_config_path)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption(
            "scheduler_config_path", scheduler_config_path
        )
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrain_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        image_process_config_path = config.getoption(
            "image_process_config_path", image_process_config_path
        )
        image_process_config_path = pop_value(
            image_process_config_path,
            nested_dict_value(pretrain_infos, "image", "vision_config"),
        )
        image_process_config_path = cached_path(image_process_config_path)

        quant_config_path = config.getoption("quant_config_path", quant_config_path)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        weight_path = config.getoption("pretrained_weight_path", pretrained_weight_path)
        device = config.getoption("device", device)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(nested_dict_value(pretrain_infos, "unet", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "image", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
            ]

        inst = cls(
            config_path=config_path,
            image_config_path=image_config_path,
            image_process_config_path=image_process_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            weight_path=weight_path,
            state_dict=state_dict,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/stable/image2video")
    def __call__(
        self,
        image: Image.Image,
        height: Optional[int] = 576,
        width: Optional[int] = 1024,
        num_frames: Optional[int] = 30,
        num_fps: Optional[int] = 6,
        min_guidance_scale: Optional[float] = 1.0,
        max_guidance_scale: Optional[float] = 2.5,
        motion_bucket_id: Optional[int] = 127,
        decode_chunk_size: Optional[int] = 8,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
    ):
        image = image.convert("RGB")
        inputs = self.processor.image2video_inputs(
            image,
            vae_image=image.resize((width, height)),
        )

        assert scheduler is None or scheduler in Schedulers
        if scheduler is not None:
            self.scheduler = Schedulers.get(scheduler).from_config(
                self.scheduler.config
            )
        self.scheduler.set_timesteps(num_inference_steps=num_timesteps)

        self.pipeline = StableVideoDiffusionPipelineV2(
            vae=self.vae,
            image_encoder=self.image,
            unet=self.unet,
            scheduler=self.scheduler,
            feature_extractor=self.image_processor,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

        if self._enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

        outputs = self.pipeline(
            image=inputs["pixel_values"],
            vae_image=inputs["vae_pixel_values"],
            height=height,
            width=width,
            num_frames=num_frames,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            fps=num_fps,
            motion_bucket_id=motion_bucket_id,
            decode_chunk_size=decode_chunk_size,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            output_type="pt",
        )

        frames = tensor2vid(outputs.frames)
        name = export_to_video(frames)
        return name
