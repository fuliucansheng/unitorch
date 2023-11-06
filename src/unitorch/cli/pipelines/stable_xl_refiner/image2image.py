# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from unitorch import is_xformers_available
from unitorch.models.diffusers import (
    StableXLRefinerForImage2ImageGeneration as _StableXLRefinerForImage2ImageGeneration,
)
from unitorch.models.diffusers import StableXLRefinerProcessor

from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.diffusers import pretrained_diffusers_infos, load_weight


class StableXLRefinerForImage2ImageGenerationPipeline(
    _StableXLRefinerForImage2ImageGeneration
):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab1_path: str,
        merge1_path: str,
        vocab2_path: str,
        merge2_path: str,
        refiner_config_path: Optional[str] = None,
        refiner_text_config_path: Optional[str] = None,
        refiner_text2_config_path: Optional[str] = None,
        refiner_vae_config_path: Optional[str] = None,
        refiner_scheduler_config_path: Optional[str] = None,
        refiner_vocab1_path: Optional[str] = None,
        refiner_merge1_path: Optional[str] = None,
        refiner_vocab2_path: Optional[str] = None,
        refiner_merge2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            refiner_config_path=refiner_config_path,
            refiner_text_config_path=refiner_text_config_path,
            refiner_text2_config_path=refiner_text2_config_path,
            refiner_vae_config_path=refiner_vae_config_path,
            refiner_scheduler_config_path=refiner_scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = StableXLRefinerProcessor(
            vocab1_path=vocab1_path,
            merge1_path=merge1_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            refiner_vocab1_path=refiner_vocab1_path,
            refiner_merge1_path=refiner_merge1_path,
            refiner_vocab2_path=refiner_vocab2_path,
            refiner_merge2_path=refiner_merge2_path,
            vae_config_path=vae_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.eval()

        if enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
            self.refiner_pipeline.enable_model_cpu_offload(self._device)
            self.to(torch.half)
        else:
            self.to(device=self._device)

        if enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()
            self.refiner_pipeline.enable_xformers_memory_efficient_attention()

    @classmethod
    @add_default_section_for_init("core/pipeline/stable_xl_refiner/image2image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/pipeline/stable_xl_refiner/image2image")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-xl-base-refiner-1.0"
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

        text2_config_path = config.getoption("text2_config_path", None)
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrain_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrain_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab1_path = config.getoption("vocab1_path", None)
        vocab1_path = pop_value(
            vocab1_path,
            nested_dict_value(pretrain_infos, "text", "vocab"),
        )
        vocab1_path = cached_path(vocab1_path)

        merge1_path = config.getoption("merge1_path", None)
        merge1_path = pop_value(
            merge1_path,
            nested_dict_value(pretrain_infos, "text", "merge"),
        )
        merge1_path = cached_path(merge1_path)

        vocab2_path = config.getoption("vocab2_path", None)
        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrain_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        merge2_path = config.getoption("merge2_path", None)
        merge2_path = pop_value(
            merge2_path,
            nested_dict_value(pretrain_infos, "text2", "merge"),
        )
        merge2_path = cached_path(merge2_path)

        refiner_config_path = config.getoption("refiner_config_path", None)
        refiner_config_path = pop_value(
            refiner_config_path,
            nested_dict_value(pretrain_infos, "refiner_unet", "config"),
            check_none=False,
        )
        refiner_config_path = (
            cached_path(refiner_config_path)
            if refiner_config_path is not None
            else None
        )

        refiner_text_config_path = config.getoption("refiner_text_config_path", None)
        refiner_text_config_path = pop_value(
            refiner_text_config_path,
            nested_dict_value(pretrain_infos, "refiner_text", "config"),
            check_none=False,
        )
        refiner_text_config_path = (
            cached_path(refiner_text_config_path)
            if refiner_text_config_path is not None
            else None
        )

        refiner_text2_config_path = config.getoption("refiner_text2_config_path", None)
        refiner_text2_config_path = pop_value(
            refiner_text2_config_path,
            nested_dict_value(pretrain_infos, "refiner_text2", "config"),
            check_none=False,
        )
        refiner_text2_config_path = (
            cached_path(refiner_text2_config_path)
            if refiner_text2_config_path is not None
            else None
        )

        refiner_vae_config_path = config.getoption("refiner_vae_config_path", None)
        refiner_vae_config_path = pop_value(
            refiner_vae_config_path,
            nested_dict_value(pretrain_infos, "refiner_vae", "config"),
        )
        refiner_vae_config_path = cached_path(refiner_vae_config_path)

        refiner_scheduler_config_path = config.getoption(
            "refiner_scheduler_config_path", None
        )
        refiner_scheduler_config_path = pop_value(
            refiner_scheduler_config_path,
            nested_dict_value(pretrain_infos, "refiner_scheduler"),
            check_none=False,
        )
        refiner_scheduler_config_path = (
            cached_path(refiner_scheduler_config_path)
            if refiner_scheduler_config_path is not None
            else None
        )

        refiner_vocab1_path = config.getoption("refiner_vocab1_path", None)
        refiner_vocab1_path = pop_value(
            refiner_vocab1_path,
            nested_dict_value(pretrain_infos, "refiner_text", "vocab"),
            check_none=False,
        )
        refiner_vocab1_path = (
            cached_path(refiner_vocab1_path) if refiner_vocab1_path else None
        )

        refiner_merge1_path = config.getoption("refiner_merge1_path", None)
        refiner_merge1_path = pop_value(
            refiner_merge1_path,
            nested_dict_value(pretrain_infos, "refiner_text", "merge"),
            check_none=False,
        )
        refiner_merge1_path = (
            cached_path(refiner_merge1_path) if refiner_merge1_path else None
        )

        refiner_vocab2_path = config.getoption("refiner_vocab2_path", None)
        refiner_vocab2_path = pop_value(
            refiner_vocab2_path,
            nested_dict_value(pretrain_infos, "refiner_text2", "vocab"),
            check_none=False,
        )
        refiner_vocab2_path = (
            cached_path(refiner_vocab2_path) if refiner_vocab2_path else None
        )

        refiner_merge2_path = config.getoption("merge2_path", None)
        refiner_merge2_path = pop_value(
            refiner_merge2_path,
            nested_dict_value(pretrain_infos, "refiner_text2", "merge"),
            check_none=False,
        )
        refiner_merge2_path = (
            cached_path(refiner_merge2_path) if refiner_merge2_path else None
        )

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        weight_path = config.getoption("pretrained_weight_path", None)
        device = config.getoption("device", "cpu")
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"), "unet."
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "text", "weight"), "text."
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "text2", "weight"), "text2."
                ),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight"), "vae."),
            ]
            if nested_dict_value(pretrain_infos, "refiner_unet", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_unet", "weight"),
                        "refiner_unet.",
                    )
                )
            if nested_dict_value(pretrain_infos, "refiner_text", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_text", "weight"),
                        "refiner_text.",
                    )
                )
            if nested_dict_value(pretrain_infos, "refiner_text2", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_text2", "weight"),
                        "refiner_text2.",
                    )
                )
            if nested_dict_value(pretrain_infos, "refiner_vae", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_vae", "weight"),
                        "refiner_vae.",
                    )
                )

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab1_path=vocab1_path,
            merge1_path=merge1_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            refiner_config_path=refiner_config_path,
            refiner_text_config_path=refiner_text_config_path,
            refiner_text2_config_path=refiner_text2_config_path,
            refiner_vae_config_path=refiner_vae_config_path,
            refiner_scheduler_config_path=refiner_scheduler_config_path,
            refiner_vocab1_path=refiner_vocab1_path,
            refiner_merge1_path=refiner_merge1_path,
            refiner_vocab2_path=refiner_vocab2_path,
            refiner_merge2_path=refiner_merge2_path,
            quant_config_path=quant_config_path,
            weight_path=weight_path,
            state_dict=state_dict,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/stable_xl_refiner/image2image")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        inputs = self.processor.image2image_inputs(text, image=image)
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        self.scheduler.set_timesteps(num_inference_steps=num_timesteps)
        self.seed = seed
        outputs = self.generate(
            **inputs,
            strength=strength,
            guidance_scale=guidance_scale,
        )
        images = numpy_to_pil(outputs.images.cpu().numpy())
        return images[0]
