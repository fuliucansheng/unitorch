# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import re
import json
import logging
import torch
import hashlib
import pandas as pd
from PIL import Image
from torch import autocast
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers.utils import numpy_to_pil
from diffusers.models import ControlNetModel
from diffusers.pipelines import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.models.diffusers import GenericStableXLModel
from unitorch.models.diffusers import StableXLProcessor

from unitorch.utils import pop_value, nested_dict_value
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


class ControlNetXLForImage2ImageFastAPIPipeline(GenericStableXLModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        merge2_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        pad_token: Optional[str] = "<|endoftext|>",
        pad_token2: Optional[str] = "!",
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
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = StableXLProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            pad_token=pad_token,
            pad_token2=pad_token2,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)

        self.eval()

        self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
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
    @add_default_section_for_init("core/fastapi/pipeline/controlnet_xl/image2image")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        pretrained_controlnet_names: Optional[Union[str, List[str]]] = None,
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        text2_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vocab2_path: Optional[str] = None,
        merge2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/controlnet_xl/image2image")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "stable-xl-base"
        )
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_names = (
            pretrained_controlnet_names
            if pretrained_controlnet_names is not None
            else config.getoption(
                "pretrained_controlnet_names", "stable-xl-controlnet-canny"
            )
        )
        if isinstance(pretrained_controlnet_names, str):
            pretrained_controlnet_names = [pretrained_controlnet_names]

        pretrained_controlnet_infos = [
            nested_dict_value(
                pretrained_stable_extensions_infos, pretrained_controlnet_name
            )
            for pretrained_controlnet_name in pretrained_controlnet_names
        ]

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "unet", "config"),
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

        text2_config_path = text2_config_path or config.getoption(
            "text2_config_path", None
        )
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrained_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        vae_config_path = vae_config_path or config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        controlnet_configs_path = config.getoption("controlnet_configs_path", None)
        if isinstance(controlnet_configs_path, str):
            controlnet_configs_path = [controlnet_configs_path]
        controlnet_configs_path = pop_value(
            controlnet_configs_path,
            [
                nested_dict_value(pretrained_controlnet_info, "controlnet", "config")
                for pretrained_controlnet_info in pretrained_controlnet_infos
            ],
        )
        controlnet_configs_path = [
            cached_path(controlnet_config_path)
            for controlnet_config_path in controlnet_configs_path
        ]

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

        vocab2_path = vocab2_path or config.getoption("vocab2_path", None)
        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrained_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        merge2_path = merge2_path or config.getoption("merge2_path", None)
        merge2_path = pop_value(
            merge2_path,
            nested_dict_value(pretrained_infos, "text2", "merge"),
        )
        merge2_path = cached_path(merge2_path)

        quant_config_path = quant_config_path or config.getoption(
            "quant_config_path", None
        )
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        pad_token = config.getoption("pad_token", "<|endoftext|>")
        pad_token2 = config.getoption("pad_token2", "!")
        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", True)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "unet", "weight"),
                    prefix_keys={"": "unet."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]
            if len(pretrained_controlnet_infos) > 1:
                for i, pretrained_controlnet_info in enumerate(
                    pretrained_controlnet_infos
                ):
                    state_dict.append(
                        load_weight(
                            nested_dict_value(
                                pretrained_controlnet_info, "controlnet", "weight"
                            ),
                            prefix_keys={"": f"controlnet.nets.{i}."},
                        )
                    )
            else:
                state_dict.append(
                    load_weight(
                        nested_dict_value(
                            pretrained_controlnet_infos[0], "controlnet", "weight"
                        ),
                        prefix_keys={"": "controlnet."},
                    )
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
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            merge2_path=merge2_path,
            quant_config_path=quant_config_path,
            pad_token=pad_token,
            pad_token2=pad_token2,
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
    @add_default_section_for_function("core/fastapi/pipeline/controlnet_xl/image2image")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        controlnet_images: Optional[List[Image.Image]] = [],
        controlnet_guidance_scales: Optional[List[float]] = [],
        neg_text: Optional[str] = "",
        guidance_scale: Optional[float] = 7.5,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        text_inputs = self.processor.text2image_inputs(
            text,
            negative_prompt=neg_text,
        )
        image_inputs = self.processor.image2image_inputs(image)
        width, height = image.size
        controlnet_images = [
            img.resize((width, height), resample=Image.LANCZOS)
            for img in controlnet_images
        ]
        assert len(controlnet_images) == len(controlnet_guidance_scales)
        assert len(controlnet_images) == self.num_controlnets
        controlnets_inputs = self.processor.controlnets_inputs(controlnet_images)
        inputs = {
            **text_inputs,
            **image_inputs,
            **{"condition_pixel_values": controlnets_inputs.pixel_values},
        }
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs.get("input_ids"),
            input2_ids=inputs.get("input2_ids"),
            negative_input_ids=inputs.get("negative_input_ids"),
            negative_input2_ids=inputs.get("negative_input2_ids"),
            attention_mask=inputs.get("attention_mask"),
            attention2_mask=inputs.get("attention2_mask"),
            negative_attention_mask=inputs.get("negative_attention_mask"),
            negative_attention2_mask=inputs.get("negative_attention2_mask"),
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        prompt_embeds = prompt_outputs.prompt_embeds
        negative_prompt_embeds = prompt_outputs.negative_prompt_embeds
        pooled_prompt_embeds = prompt_outputs.pooled_prompt_embeds
        negative_pooled_prompt_embeds = prompt_outputs.negative_pooled_prompt_embeds

        outputs = self.pipeline(
            image=inputs["pixel_values"],
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            control_image=list(inputs["condition_pixel_values"].transpose(0, 1)),
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=num_timesteps,
            guidance_scale=guidance_scale,
            strength=strength,
            output_type="np.array",
        )

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0]
