# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast

from unitorch.models.diffusers import (
    MultiControlNetForText2ImageGeneration as _MultiControlNetForText2ImageGeneration,
    MultiControlNetForImage2ImageGeneration as _MultiControlNetForImage2ImageGeneration,
    MultiControlNetForImageInpainting as _MultiControlNetForImageInpainting,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models.diffusers import (
    pretrained_diffusers_infos,
    load_weight,
)


@register_model(
    "core/model/diffusers/text2image/multicontrolnet", diffusion_model_decorator
)
class MultiControlNetForText2ImageGeneration(_MultiControlNetForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: List[str],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/text2image/multicontrolnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/text2image/multicontrolnet")
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

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", True)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        freeze_unet_encoder = config.getoption("freeze_unet_encoder", True)
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            controlnets = nested_dict_value(pretrain_infos, "controlnet", "weight")
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                ),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
            ] + [
                load_weight(
                    controlnet,
                    prefix_keys={"": f"controlnet.nets.{i}."},
                )
                for i, controlnet in enumerate(controlnets)
            ]

        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        loss = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            condition_pixel_values=condition_pixel_values,
            attention_mask=attention_mask,
        )
        return LossOutputs(loss=loss)

    @add_default_section_for_function("core/model/diffusers/text2image/multicontrolnet")
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[float] = None,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            condition_pixel_values=condition_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        return DiffusionOutputs(outputs=outputs.images)


@register_model(
    "core/model/diffusers/image2image/multicontrolnet", diffusion_model_decorator
)
class MultiControlNetForImage2ImageGeneration(_MultiControlNetForImage2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: List[str],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/image2image/multicontrolnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/image2image/multicontrolnet")
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

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", True)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        freeze_unet_encoder = config.getoption("freeze_unet_encoder", True)
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            controlnets = nested_dict_value(pretrain_infos, "controlnet", "weight")
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                ),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
            ] + [
                load_weight(
                    controlnet,
                    prefix_keys={"": f"controlnet.nets.{i}."},
                )
                for i, controlnet in enumerate(controlnets)
            ]
        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function(
        "core/model/diffusers/image2image/multicontrolnet"
    )
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[float] = None,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            pixel_values=pixel_values,
            condition_pixel_values=condition_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        return DiffusionOutputs(outputs=outputs.images)


@register_model(
    "core/model/diffusers/inpainting/multicontrolnet", diffusion_model_decorator
)
class MultiControlNetForImageInpainting(_MultiControlNetForImageInpainting):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: List[str],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/inpainting/multicontrolnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/inpainting/multicontrolnet")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-v1.5-multicontrolnet-inpainting"
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

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", True)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        freeze_unet_encoder = config.getoption("freeze_unet_encoder", True)
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            controlnets = nested_dict_value(pretrain_infos, "controlnet", "weight")
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                ),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
            ] + [
                load_weight(
                    controlnet,
                    prefix_keys={"": f"controlnet.nets.{i}."},
                )
                for i, controlnet in enumerate(controlnets)
            ]
        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("core/model/diffusers/inpainting/multicontrolnet")
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_masks: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[float] = None,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            pixel_values=pixel_values,
            pixel_masks=pixel_masks,
            condition_pixel_values=condition_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        return DiffusionOutputs(outputs=outputs.images)
