# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from torch import autocast
from typing import List, Optional, Tuple, Union

from unitorch.models.peft.diffusers import (
    Flux2LoraForImageEditingGeneration as _Flux2LoraForImageEditingGeneration,
    Flux2LoraForText2ImageGeneration as _Flux2LoraForText2ImageGeneration,
)
from unitorch.utils import is_bfloat16_available, nested_dict_value
from unitorch.cli import config_defaults_init, config_defaults_method, register_model
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models.diffusers import load_weight
from unitorch.cli.models.diffusers.modeling_flux2 import _flux2_model_kwargs


_DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_qkv_mlp",
    "q_proj",
    "k_proj",
    "v_proj",
]

_DEFAULT_REPLACE_KEYS = {
    r"to_q\.": "to_q.base_layer.",
    r"to_k\.": "to_k.base_layer.",
    r"to_v\.": "to_v.base_layer.",
    r"add_q_proj\.": "add_q_proj.base_layer.",
    r"add_k_proj\.": "add_k_proj.base_layer.",
    r"add_v_proj\.": "add_v_proj.base_layer.",
    r"to_qkv_mlp\.": "to_qkv_mlp.base_layer.",
    r"\.q_proj\.": ".q_proj.base_layer.",
    r"\.k_proj\.": ".k_proj.base_layer.",
    r"\.v_proj\.": ".v_proj.base_layer.",
}


def _append_state_dict(state_dict, new_state_dict):
    if not new_state_dict:
        return state_dict
    if state_dict is None:
        return new_state_dict
    if isinstance(state_dict, list):
        return state_dict + [new_state_dict]
    return [state_dict, new_state_dict]


def _flux2_lora_model_kwargs(config, section: str):
    model_kwargs = _flux2_model_kwargs(config, section)
    model_kwargs.pop("freeze_vae_encoder", None)
    model_kwargs.pop("freeze_text_encoder", None)
    model_kwargs.update(
        {
            "lora_r": config.getoption("lora_r", 16),
            "lora_alpha": config.getoption("lora_alpha", 32),
            "lora_dropout": config.getoption("lora_dropout", 0.05),
            "fan_in_fan_out": config.getoption("fan_in_fan_out", True),
            "target_modules": config.getoption(
                "target_modules",
                _DEFAULT_TARGET_MODULES,
            ),
            "enable_text_adapter": config.getoption("enable_text_adapter", True),
            "enable_transformer_adapter": config.getoption(
                "enable_transformer_adapter",
                True,
            ),
        }
    )
    replace_keys = config.getoption("replace_keys", _DEFAULT_REPLACE_KEYS)
    return model_kwargs, replace_keys


def _flux2_lora_state_dict(
    config,
    pretrained_infos,
    use_auth_token,
    replace_keys,
    enable_text_adapter: bool,
    enable_transformer_adapter: bool,
):
    weight_path = config.getoption("pretrained_weight_path", None)

    state_dict = None
    if weight_path is None and pretrained_infos is not None:
        state_dict = [
            load_weight(
                nested_dict_value(pretrained_infos, "transformer", "weight"),
                prefix_keys={"": "transformer."},
                replace_keys=replace_keys if enable_transformer_adapter else {},
                use_auth_token=use_auth_token,
            ),
            load_weight(
                nested_dict_value(pretrained_infos, "text", "weight"),
                prefix_keys={"": "text."},
                replace_keys=replace_keys if enable_text_adapter else {},
                use_auth_token=use_auth_token,
            ),
            load_weight(
                nested_dict_value(pretrained_infos, "vae", "weight"),
                prefix_keys={"": "vae."},
                use_auth_token=use_auth_token,
            ),
        ]
    elif weight_path is not None:
        state_dict = load_weight(weight_path, use_auth_token=use_auth_token)

    pretrained_lora_weight_path = config.getoption("pretrained_lora_weight_path", None)
    if pretrained_lora_weight_path is not None:
        lora_state_dict = load_weight(
            pretrained_lora_weight_path,
            use_auth_token=use_auth_token,
        )
        state_dict = _append_state_dict(state_dict, lora_state_dict)

    return state_dict


@register_model(
    "core/model/diffusers/peft/lora/text2image/flux2", diffusion_model_decorator
)
class Flux2LoraForText2ImageGeneration(_Flux2LoraForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = _DEFAULT_TARGET_MODULES,
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 4.0,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = (10, 20, 30),
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
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
            text_encoder_out_layers=text_encoder_out_layers,
        )

    @classmethod
    @config_defaults_init("core/model/diffusers/peft/lora/text2image/flux2")
    def from_config(cls, config, **kwargs):
        section = "core/model/diffusers/peft/lora/text2image/flux2"
        model_kwargs, replace_keys = _flux2_lora_model_kwargs(config, section)
        pretrained_infos = model_kwargs.pop("pretrained_infos")
        use_auth_token = model_kwargs.pop("use_auth_token")

        inst = cls(**model_kwargs)
        state_dict = _flux2_lora_state_dict(
            config,
            pretrained_infos,
            use_auth_token,
            replace_keys,
            model_kwargs["enable_text_adapter"],
            model_kwargs["enable_transformer_adapter"],
        )
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

    @config_defaults_method("core/model/diffusers/peft/lora/text2image/flux2")
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 4.0,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
        return DiffusionOutputs(outputs=outputs.images.float())


@register_model(
    "core/model/diffusers/peft/lora/editing/flux2", diffusion_model_decorator
)
class Flux2LoraForImageEditingGeneration(_Flux2LoraForImageEditingGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = _DEFAULT_TARGET_MODULES,
        enable_text_adapter: Optional[bool] = True,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = 4.0,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = (10, 20, 30),
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
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
            text_encoder_out_layers=text_encoder_out_layers,
        )

    @classmethod
    @config_defaults_init("core/model/diffusers/peft/lora/editing/flux2")
    def from_config(cls, config, **kwargs):
        section = "core/model/diffusers/peft/lora/editing/flux2"
        model_kwargs, replace_keys = _flux2_lora_model_kwargs(config, section)
        pretrained_infos = model_kwargs.pop("pretrained_infos")
        use_auth_token = model_kwargs.pop("use_auth_token")

        inst = cls(**model_kwargs)
        state_dict = _flux2_lora_state_dict(
            config,
            pretrained_infos,
            use_auth_token,
            replace_keys,
            model_kwargs["enable_text_adapter"],
            model_kwargs["enable_transformer_adapter"],
        )
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
        attention_mask: Optional[torch.Tensor] = None,
    ):
        loss = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            refer_pixel_values=refer_pixel_values,
            attention_mask=attention_mask,
        )
        return LossOutputs(loss=loss)

    @config_defaults_method("core/model/diffusers/peft/lora/editing/flux2")
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 4.0,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            refer_pixel_values=refer_pixel_values,
            attention_mask=attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
        return DiffusionOutputs(outputs=outputs.images.float())
