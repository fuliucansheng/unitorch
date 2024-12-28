# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast

from unitorch.models.peft.diffusers import (
    ControlNetXLLoraForText2ImageGeneration as _ControlNetXLLoraForText2ImageGeneration,
)
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
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
    "core/model/diffusers/peft/lora/text2image/controlnet_xl", diffusion_model_decorator
)
class ControlNetXLLoraForText2ImageGeneration(_ControlNetXLLoraForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "q_proj",
            "v_proj",
            "out_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_unet_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        use_fp16: Optional[bool] = True,
        use_bf16: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_unet_adapter=enable_unet_adapter,
            seed=seed,
        )
        self.use_dtype = torch.float16 if use_fp16 else torch.float32
        self.use_dtype = (
            torch.bfloat16 if use_bf16 and is_bfloat16_available() else self.use_dtype
        )

    @classmethod
    @add_default_section_for_init(
        "core/model/diffusers/peft/lora/text2image/controlnet_xl"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "core/model/diffusers/peft/lora/text2image/controlnet_xl"
        )
        pretrained_name = config.getoption("pretrained_name", "stable-xl-base")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        text2_config_path = config.getoption("text2_config_path", None)
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrained_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        pretrained_controlnet_name = config.getoption(
            "pretrained_controlnet_name", "stable-xl-controlnet-canny"
        )
        pretrained_controlnet_infos = nested_dict_value(
            pretrained_stable_extensions_infos, pretrained_controlnet_name
        )

        controlnet_configs_path = config.getoption("controlnet_configs_path", None)
        controlnet_configs_path = pop_value(
            controlnet_configs_path,
            nested_dict_value(pretrained_controlnet_infos, "controlnet", "config"),
        )
        controlnet_configs_path = cached_path(controlnet_configs_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
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
        snr_gamma = config.getoption("snr_gamma", 5.0)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption(
            "target_modules",
            [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "q_proj",
                "v_proj",
                "out_proj",
            ],
        )
        replace_keys = config.getoption(
            "replace_keys",
            {
                "to_k.": "to_k.base_layer.",
                "to_q.": "to_q.base_layer.",
                "to_v.": "to_v.base_layer.",
                "to_out.0.": "to_out.0.base_layer.",
                "q_proj.": "q_proj.base_layer.",
                "v_proj.": "v_proj.base_layer.",
                "out_proj.": "out_proj.base_layer.",
            },
        )
        enable_text_adapter = config.getoption("enable_text_adapter", True)
        enable_unet_adapter = config.getoption("enable_unet_adapter", True)

        seed = config.getoption("seed", 1123)
        use_fp16 = config.getoption("use_fp16", True)
        use_bf16 = config.getoption("use_bf16", False)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_unet_adapter=enable_unet_adapter,
            seed=seed,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "unet", "weight"),
                    prefix_keys={"": "unet."},
                    replace_keys=replace_keys if enable_unet_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                    replace_keys=replace_keys if enable_text_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                    replace_keys=replace_keys if enable_text_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
                load_weight(
                    nested_dict_value(
                        pretrained_controlnet_infos, "controlnet", "weight"
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

    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        add_time_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
    ):
        with autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=self.use_dtype,
        ):
            loss = super().forward(
                input_ids=input_ids,
                input2_ids=input2_ids,
                add_time_ids=add_time_ids,
                pixel_values=pixel_values,
                condition_pixel_values=condition_pixel_values,
                attention_mask=attention_mask,
                attention2_mask=attention2_mask,
            )
            return LossOutputs(loss=loss)

    @add_default_section_for_function(
        "core/model/diffusers/peft/lora/text2image/controlnet_xl"
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[float] = 0.5,
    ):
        with autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=self.use_dtype,
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
