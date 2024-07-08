# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast

from unitorch.models.peft.diffusers import (
    ControlNetLoraForText2ImageGeneration as _ControlNetLoraForText2ImageGeneration,
    ControlNetLoraForImage2ImageGeneration as _ControlNetLoraForImage2ImageGeneration,
    ControlNetLoraForImageInpainting as _ControlNetLoraForImageInpainting,
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
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)


@register_model(
    "core/model/peft/diffusers/text2image/controlnet", diffusion_model_decorator
)
class ControlNetLoraForText2ImageGeneration(_ControlNetLoraForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        lora_r: Optional[int] = 16,
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
            lora_r=lora_r,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/peft/diffusers/text2image/controlnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/peft/diffusers/text2image/controlnet")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
        pretrain_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_name = config.getoption(
            "pretrained_controlnet_name", "stable-v1.5-controlnet-canny"
        )
        pretrain_controlnet_infos = nested_dict_value(
            pretrained_stable_extensions_infos, pretrained_controlnet_name
        )

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
            nested_dict_value(pretrain_controlnet_infos, "controlnet", "config"),
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

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        lora_r = config.getoption("lora_r", 16)
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
            lora_r=lora_r,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                    replace_keys={
                        "to_k.": "to_k.base_layer.",
                        "to_q.": "to_q.base_layer.",
                        "to_v.": "to_v.base_layer.",
                        "to_out.0.": "to_out.0.base_layer.",
                    },
                ),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
                load_weight(
                    nested_dict_value(
                        pretrain_controlnet_infos, "controlnet", "weight"
                    ),
                    prefix_keys={"": "controlnet."},
                ),
            ]
        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            lora_state_dict = load_weight(pretrained_lora_weight_path)
            if state_dict is not None:
                state_dict.append(lora_state_dict)
            else:
                state_dict = lora_state_dict

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

    @add_default_section_for_function("core/model/peft/diffusers/text2image/controlnet")
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
        controlnet_conditioning_scale: Optional[float] = 1.0,
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
    "core/model/peft/diffusers/image2image/controlnet", diffusion_model_decorator
)
class ControlNetLoraForImage2ImageGeneration(_ControlNetLoraForImage2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        lora_r: Optional[int] = 16,
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
            lora_r=lora_r,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/peft/diffusers/image2image/controlnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/peft/diffusers/image2image/controlnet")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
        pretrain_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_name = config.getoption(
            "pretrained_controlnet_name", "stable-v1.5-controlnet-canny"
        )
        pretrain_controlnet_infos = nested_dict_value(
            pretrained_stable_extensions_infos, pretrained_controlnet_name
        )

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
            nested_dict_value(pretrain_controlnet_infos, "controlnet", "config"),
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

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        lora_r = config.getoption("lora_r", 16)
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
            lora_r=lora_r,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                    replace_keys={
                        "to_k.": "to_k.base_layer.",
                        "to_q.": "to_q.base_layer.",
                        "to_v.": "to_v.base_layer.",
                        "to_out.0.": "to_out.0.base_layer.",
                    },
                ),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
                load_weight(
                    nested_dict_value(
                        pretrain_controlnet_infos, "controlnet", "weight"
                    ),
                    prefix_keys={"": "controlnet."},
                ),
            ]
        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            lora_state_dict = load_weight(pretrained_lora_weight_path)
            if state_dict is not None:
                state_dict.append(lora_state_dict)
            else:
                state_dict = lora_state_dict

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function(
        "core/model/peft/diffusers/image2image/controlnet"
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
        controlnet_conditioning_scale: Optional[float] = 1.0,
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
    "core/model/peft/diffusers/inpainting/controlnet", diffusion_model_decorator
)
class ControlNetLoraForImageInpainting(_ControlNetLoraForImageInpainting):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        controlnet_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        lora_r: Optional[int] = 16,
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
            lora_r=lora_r,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/peft/diffusers/inpainting/controlnet")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/peft/diffusers/inpainting/controlnet")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
        pretrain_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_name = config.getoption(
            "pretrained_controlnet_name", "stable-v1.5-controlnet-canny"
        )
        pretrain_controlnet_infos = nested_dict_value(
            pretrained_stable_extensions_infos, pretrained_controlnet_name
        )

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
            nested_dict_value(pretrain_controlnet_infos, "controlnet", "config"),
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

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        lora_r = config.getoption("lora_r", 16)
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
            lora_r=lora_r,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                    replace_keys={
                        "to_k.": "to_k.base_layer.",
                        "to_q.": "to_q.base_layer.",
                        "to_v.": "to_v.base_layer.",
                        "to_out.0.": "to_out.0.base_layer.",
                    },
                ),
                load_weight(nested_dict_value(pretrain_infos, "text", "weight")),
                load_weight(nested_dict_value(pretrain_infos, "vae", "weight")),
                load_weight(
                    nested_dict_value(
                        pretrain_controlnet_infos, "controlnet", "weight"
                    ),
                    prefix_keys={"": "controlnet."},
                ),
            ]
        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            lora_state_dict = load_weight(pretrained_lora_weight_path)
            if state_dict is not None:
                state_dict.append(lora_state_dict)
            else:
                state_dict = lora_state_dict

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("core/model/peft/diffusers/inpainting/controlnet")
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
        controlnet_conditioning_scale: Optional[float] = 1.0,
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
