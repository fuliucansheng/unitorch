# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from unitorch import is_xformers_available
from unitorch.models.diffusers import (
    MultiControlNetForImageInpainting as _MultiControlNetForImageInpainting,
)
from unitorch.models.diffusers import MultiControlNetProcessor

from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.diffusers import pretrained_diffusers_infos, load_weight
from unitorch.cli.pipelines import Schedulers


class MultiControlNetForImageInpaintingPipeline(_MultiControlNetForImageInpainting):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: List[str],
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
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = MultiControlNetProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            pad_token=pad_token,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        if enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
            self.to(torch.half)
        else:
            self.to(device=self._device)

        if enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

    @classmethod
    @add_default_section_for_init("core/pipeline/multicontrolnet/inpainting")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/pipeline/multicontrolnet/inpainting")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-v1.5-multicontrolnet-canny-depth"
        )
        pretrain_infos = nested_dict_value(pretrained_diffusers_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrain_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrain_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        controlnet_configs_path = config.getoption("controlnet_configs_path", None)
        controlnet_configs_path = pop_value(
            controlnet_configs_path,
            nested_dict_value(pretrain_infos, "controlnet", "config"),
        )
        if isinstance(controlnet_configs_path, str):
            controlnet_configs_path = [controlnet_configs_path]
        else:
            controlnet_configs_path = [cached_path(p) for p in controlnet_configs_path]

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrain_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrain_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrain_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        pad_token = config.getoption("pad_token", "<|endoftext|>")
        weight_path = config.getoption("pretrained_weight_path", None)
        device = config.getoption("device", "cpu")
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            controlnets = nested_dict_value(pretrain_infos, "controlnet", "weight")
            state_dict = [
                load_weight(nested_dict_value(pretrain_infos, "unet", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
            ] + [
                load_weight(
                    controlnet,
                    prefix_keys={"": f"controlnet.nets.{i}."},
                )
                for i, controlnet in enumerate(controlnets)
            ]

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
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
    @add_default_section_for_function("core/pipeline/multicontrolnet/inpainting")
    def __call__(
        self,
        text: str,
        image: Union[Image.Image, str],
        mask_image: Union[Image.Image, str],
        condition_images: List[Union[Image.Image, str]],
        neg_text: Optional[str] = "",
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[List[float]] = None,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        scheduler: Optional[str] = None,
        freeu_params: Optional[Tuple[float, float, float, float]] = (
            0.9,
            0.2,
            1.2,
            1.4,
        ),
    ):
        inputs = self.processor.inpainting_inputs(
            image=image,
            mask_image=mask_image,
            condition_images=condition_images,
            prompt=text,
            negative_prompt=neg_text,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        assert scheduler is None or scheduler in Schedulers
        if scheduler is not None:
            self.scheduler = Schedulers.get(scheduler).from_config(
                self.scheduler.config
            )
        self.scheduler.set_timesteps(num_inference_steps=num_timesteps)
        self.pipeline.enable_freeu(*freeu_params)
        self.seed = seed
        outputs = self.generate(
            **inputs,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        images = numpy_to_pil(outputs.images.cpu().numpy())
        return images[0]
