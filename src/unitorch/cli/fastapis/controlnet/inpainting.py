# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import io
import re
import json
import logging
import torch
import hashlib
import pandas as pd
from torch import autocast
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
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


class ControlNetForImageInpaintingFastAPIPipeline(GenericStableModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        inpainting_controlnet_config_path: Union[str] = None,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        pad_token: Optional[str] = "<|endoftext|>",
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
            controlnet_configs_path=controlnet_configs_path,
            inpainting_controlnet_config_path=inpainting_controlnet_config_path,
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

        self.pipeline = StableDiffusionControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
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
    @add_default_section_for_init("core/fastapi/pipeline/controlnet/inpainting")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "stable-v1.5",
        pretrained_controlnet_names: Optional[
            Union[str, List[str]]
        ] = "stable-v1.5-controlnet-canny",
        pretrained_inpainting_controlnet_name: Optional[str] = None,
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = 1.0,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = 32.0,
        **kwargs,
    ):
        config.set_default_section("core/fastapi/pipeline/controlnet/inpainting")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_names = config.getoption(
            "pretrained_controlnet_names", pretrained_controlnet_names
        )
        if isinstance(pretrained_controlnet_names, str):
            pretrained_controlnet_names = [pretrained_controlnet_names]

        pretrained_controlnet_infos = [
            nested_dict_value(
                pretrained_stable_extensions_infos, pretrained_controlnet_name
            )
            for pretrained_controlnet_name in pretrained_controlnet_names
        ]

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

        pretrained_inpainting_controlnet_name = config.getoption(
            "pretrained_inpainting_controlnet_name",
            pretrained_inpainting_controlnet_name,
        )
        inpainting_controlnet_config_path = config.getoption(
            "inpainting_controlnet_config_path", None
        )
        inpainting_controlnet_config_path = pop_value(
            inpainting_controlnet_config_path,
            nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_inpainting_controlnet_name,
                "controlnet",
                "config",
            ),
            check_none=False,
        )
        inpainting_controlnet_config_path = (
            cached_path(inpainting_controlnet_config_path)
            if inpainting_controlnet_config_path is not None
            else None
        )
        if pretrained_inpainting_controlnet_name is not None:
            pretrained_controlnet_infos.append(
                nested_dict_value(
                    pretrained_stable_extensions_infos,
                    pretrained_inpainting_controlnet_name,
                )
            )

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

        pretrained_lora_names = config.getoption(
            "pretrained_lora_names", pretrained_lora_names
        )
        pretrained_lora_weights = config.getoption(
            "pretrained_lora_weights", pretrained_lora_weights
        )
        pretrained_lora_alphas = config.getoption(
            "pretrained_lora_alphas", pretrained_lora_alphas
        )

        if isinstance(pretrained_lora_names, str):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif isinstance(pretrained_lora_names, list):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos, name, "lora", "weight"
                )
                for name in pretrained_lora_names
            ]
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_weights)
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_alphas)
        else:
            pretrained_lora_weights_path = None

        lora_weights_path = config.getoption(
            "pretrained_lora_weights_path", pretrained_lora_weights_path
        )

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            inpainting_controlnet_config_path=inpainting_controlnet_config_path,
            quant_config_path=quant_config_path,
            pad_token=pad_token,
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
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    @add_default_section_for_function("core/fastapi/pipeline/controlnet/inpainting")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
        controlnet_images: Optional[List[Image.Image]] = [],
        controlnet_guidance_scales: Optional[List[float]] = [],
        inpaint_controlnet_image: Optional[Image.Image] = None,
        inpaint_controlnet_guidance_scale: Optional[float] = 0.8,
        neg_text: Optional[str] = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = 7.5,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
        freeu_params: Optional[Tuple[float, float, float, float]] = (
            0.9,
            0.2,
            1.2,
            1.4,
        ),
    ):
        if width is None or height is None:
            width, height = image.size
        width = width // 8 * 8
        height = height // 8 * 8
        text_inputs = self.processor.text2image_inputs(
            text,
            negative_prompt=neg_text,
        )
        image_inputs = self.processor.inpainting_inputs(image, mask_image)
        if len(controlnet_images) > 0:
            controlnet_images = [
                img.resize((width, height)) for img in controlnet_images
            ]
            controlnets_inputs = self.processor.controlnets_inputs(controlnet_images)
        if inpaint_controlnet_image is not None:
            inpaint_controlnet_image = inpaint_controlnet_image.resize((width, height))
            inpaint_controlnet_inputs = self.processor.inpainting_control_inputs(
                inpaint_controlnet_image, mask_image
            )
            if len(controlnet_images) > 0:
                condition_pixel_values = torch.cat(
                    [
                        controlnets_inputs.pixel_values,
                        inpaint_controlnet_inputs.pixel_values.unsqueeze(0),
                    ],
                    dim=0,
                )
                controlnet_guidance_scales.append(inpaint_controlnet_guidance_scale)
            else:
                condition_pixel_values = (
                    inpaint_controlnet_inputs.pixel_values.unsqueeze(0)
                )
                controlnet_guidance_scales = [inpaint_controlnet_guidance_scale]
        else:
            condition_pixel_values = controlnets_inputs.pixel_values
        assert condition_pixel_values.size(0) == self.num_controlnets
        if self.num_controlnets == 1:
            controlnet_guidance_scales = controlnet_guidance_scales[0]
        inputs = {
            **text_inputs,
            **image_inputs,
            **{"condition_pixel_values": condition_pixel_values},
        }
        # if freeu_params is not None:
        #     self.pipeline.enable_freeu(*freeu_params)
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

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

        outputs = self.pipeline(
            image=inputs["pixel_values"],
            mask_image=inputs["pixel_masks"],
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            control_image=list(inputs["condition_pixel_values"].transpose(0, 1)),
            height=height,
            width=width,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=num_timesteps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_guidance_scales,
            strength=strength,
            output_type="np.array",
        )

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0]
