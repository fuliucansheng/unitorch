# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast

from unitorch.models.peft.diffusers import (
    QWenImageLoraForText2ImageGeneration as _QWenImageLoraForText2ImageGeneration,
    QWenImageLoraForImageEditing as _QWenImageLoraForImageEditing,
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
    "core/model/diffusers/peft/lora/text2image/qwen_image", diffusion_model_decorator
)
class QWenImageLoraForText2ImageGeneration(_QWenImageLoraForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 3.5,
        prompt_start_index: Optional[int] = 34,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            guidance_scale=guidance_scale,
            prompt_start_index=prompt_start_index,
        )

    @classmethod
    @add_default_section_for_init(
        "core/model/diffusers/peft/lora/text2image/qwen_image"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "core/model/diffusers/peft/lora/text2image/qwen_image"
        )
        pretrained_name = config.getoption("pretrained_name", "qwen-image")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

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
                "to_q",
                "to_k",
                "to_v",
                "q_proj",
                "k_proj",
                "v_proj",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
            ],
        )
        replace_keys = config.getoption(
            "replace_keys",
            {
                "to_q.": "to_q.base_layer.",
                "to_k.": "to_k.base_layer.",
                "to_v.": "to_v.base_layer.",
                "\.q_proj.": ".q_proj.base_layer.",
                "\.k_proj.": ".k_proj.base_layer.",
                "\.v_proj.": ".v_proj.base_layer.",
                "add_q_proj.": "add_q_proj.base_layer.",
                "add_k_proj.": "add_k_proj.base_layer.",
                "add_v_proj.": "add_v_proj.base_layer.",
            },
        )
        enable_text_adapter = config.getoption("enable_text_adapter", True)
        enable_transformer_adapter = config.getoption(
            "enable_transformer_adapter", True
        )
        seed = config.getoption("seed", 1123)
        gradient_checkpointing = config.getoption("gradient_checkpointing", True)
        guidance_scale = config.getoption("guidance_scale", 3.5)
        prompt_start_index = config.getoption("prompt_start_index", 34)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            guidance_scale=guidance_scale,
            prompt_start_index=prompt_start_index,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                    replace_keys=replace_keys if enable_transformer_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={
                        "^visual.": "text.model.",
                        "^model(?!\.model).": "text.model.language_",
                        "^lm_head.": "text.",
                    },
                    replace_keys=replace_keys if enable_text_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
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

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        loss = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        return LossOutputs(loss=loss)

    @add_default_section_for_function(
        "core/model/diffusers/peft/lora/text2image/qwen_image"
    )
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 1.0,
        true_cfg_scale: Optional[float] = 4.0,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
        )

        return DiffusionOutputs(outputs=outputs.images)


@register_model(
    "core/model/diffusers/peft/lora/editing/qwen_image", diffusion_model_decorator
)
class QWenImageLoraForImageEditing(_QWenImageLoraForImageEditing):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ],
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 3.5,
        prompt_start_index: Optional[int] = 64,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            guidance_scale=guidance_scale,
            prompt_start_index=prompt_start_index,
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/peft/lora/editing/qwen_image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/peft/lora/editing/qwen_image")
        pretrained_name = config.getoption("pretrained_name", "qwen-image-editing")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

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
                "to_q",
                "to_k",
                "to_v",
                "q_proj",
                "k_proj",
                "v_proj",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
            ],
        )
        replace_keys = config.getoption(
            "replace_keys",
            {
                "to_q.": "to_q.base_layer.",
                "to_k.": "to_k.base_layer.",
                "to_v.": "to_v.base_layer.",
                "\.q_proj.": ".q_proj.base_layer.",
                "\.k_proj.": ".k_proj.base_layer.",
                "\.v_proj.": ".v_proj.base_layer.",
                "add_q_proj.": "add_q_proj.base_layer.",
                "add_k_proj.": "add_k_proj.base_layer.",
                "add_v_proj.": "add_v_proj.base_layer.",
            },
        )
        enable_text_adapter = config.getoption("enable_text_adapter", True)
        enable_transformer_adapter = config.getoption(
            "enable_transformer_adapter", True
        )
        seed = config.getoption("seed", 1123)
        gradient_checkpointing = config.getoption("gradient_checkpointing", True)
        guidance_scale = config.getoption("guidance_scale", 3.5)
        prompt_start_index = config.getoption("prompt_start_index", 64)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            snr_gamma=snr_gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            guidance_scale=guidance_scale,
            prompt_start_index=prompt_start_index,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                    replace_keys=replace_keys if enable_transformer_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={
                        "^visual.": "text.model.",
                        "^model(?!\.model).": "text.model.language_",
                        "^lm_head.": "text.",
                    },
                    replace_keys=replace_keys if enable_text_adapter else {},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
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

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        refer_image_grid_thw: torch.Tensor,
        refer_vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        loss = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            refer_pixel_values=refer_pixel_values,
            refer_image_grid_thw=refer_image_grid_thw,
            refer_vae_pixel_values=refer_vae_pixel_values,
            attention_mask=attention_mask,
        )
        return LossOutputs(loss=loss)

    @add_default_section_for_function(
        "core/model/diffusers/peft/lora/editing/qwen_image"
    )
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        refer_image_grid_thw: torch.Tensor,
        refer_vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 1.0,
        true_cfg_scale: Optional[float] = 4.0,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            refer_pixel_values=refer_pixel_values,
            refer_image_grid_thw=refer_image_grid_thw,
            refer_vae_pixel_values=refer_vae_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
        )

        return DiffusionOutputs(outputs=outputs.images)
