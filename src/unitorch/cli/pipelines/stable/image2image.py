# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from unitorch.models.diffusers import (
    StableForImage2ImageGeneration as _StableForImage2ImageGeneration,
)
from unitorch.models.diffusers import StableProcessor

from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.diffusers import pretrained_diffusers_infos


class StableForImage2ImageGenerationPipeline(_StableForImage2ImageGeneration):
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
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("core/pipeline/stable/image2image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/pipeline/stable/image2image")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
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

        if weight_path is None and pretrain_infos is not None:
            weight_path = [
                cached_path(nested_dict_value(pretrain_infos, "unet", "weight")),
                cached_path(nested_dict_value(pretrain_infos, "text", "weight")),
                cached_path(nested_dict_value(pretrain_infos, "vae", "weight")),
            ]

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            quant_config_path=quant_config_path,
            max_seq_length=max_seq_length,
            pad_token=pad_token,
            weight_path=weight_path,
            device=device,
        )
        return inst

    @torch.no_grad()
    @add_default_section_for_function("core/pipeline/stable/image2image")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        num_timesteps: Optional[int] = 50,
    ):
        inputs = self.processor.image2image_inputs(text, image=image)
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        self.scheduler.set_timesteps(num_inference_steps=num_timesteps)
        outputs = self.generate(
            **inputs,
            strength=strength,
            guidance_scale=guidance_scale,
        )
        images = numpy_to_pil(outputs.images.cpu().numpy())
        return images[0]
