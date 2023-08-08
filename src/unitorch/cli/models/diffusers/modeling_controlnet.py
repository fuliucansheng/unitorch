# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast

from unitorch.models.diffusers import (
    ControlNetForImageGeneration as _ControlNetForImageGeneration,
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
from unitorch.cli.models.diffusers import pretrained_diffusers_infos


@register_model("core/model/diffusion/controlnet", diffusion_model_decorator)
class ControlNetForImageGeneration(_ControlNetForImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = 224,
        in_channels: Optional[int] = 4,
        out_channels: Optional[int] = 4,
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
            controlnet_config_path=controlnet_config_path,
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
    @add_default_section_for_init("core/model/diffusion/controlnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusion/controlnet")
        pretrained_name = config.getoption("pretrained_name", "controlnet-canny")
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

        controlnet_config_path = config.getoption("controlnet_config_path", None)
        controlnet_config_path = pop_value(
            controlnet_config_path,
            nested_dict_value(pretrain_infos, "controlnet", "config"),
        )
        controlnet_config_path = cached_path(controlnet_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrain_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        image_size = config.getoption("image_size", 224)
        in_channels = config.getoption("in_channels", 4)
        out_channels = config.getoption("out_channels", 4)
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
            controlnet_config_path=controlnet_config_path,
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
            weight_path = [
                cached_path(nested_dict_value(pretrain_infos, "unet", "weight")),
                cached_path(nested_dict_value(pretrain_infos, "text", "weight")),
                cached_path(nested_dict_value(pretrain_infos, "vae", "weight")),
                # cached_path(nested_dict_value(pretrain_infos, "controlnet", "weight")),
            ]
            controlnet = cached_path(
                nested_dict_value(pretrain_infos, "controlnet", "weight")
            )
            state_dict = torch.load(controlnet, map_location="cpu")
            state_dict = {f"controlnet.{k}": v for k, v in state_dict.items()}

        if weight_path is not None:
            inst.from_pretrained(weight_path, state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = super().forward(
            pixel_values=pixel_values,
            condition_pixel_values=condition_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return LossOutputs(loss=outputs)

    @add_default_section_for_function("core/model/diffusion/controlnet")
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 7.5,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            condition_pixel_values=condition_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            guidance_scale=guidance_scale,
        )

        return DiffusionOutputs(outputs=outputs.images)
