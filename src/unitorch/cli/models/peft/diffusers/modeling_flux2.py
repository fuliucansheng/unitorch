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
from unitorch.cli.models.diffusers.modeling_flux2 import _flux2_model_kwargs, _flux2_state_dict


# Target names and PEFT base-layer mapping belong to the package model. In
# particular the fused projection is to_qkv_mlp_proj in current Diffusers.
_DEFAULT_TARGET_MODULES = None
_DEFAULT_REPLACE_KEYS = {}


def _flux2_lora_model_kwargs(config, section: str):
    model_kwargs = _flux2_model_kwargs(config, section)
    model_kwargs.pop("freeze_vae_encoder", None)
    model_kwargs.pop("freeze_text_encoder", None)
    model_kwargs.update(
        {
            "lora_r": config.getoption("lora_r", 16),
            "lora_alpha": config.getoption("lora_alpha", 32),
            "lora_dropout": config.getoption("lora_dropout", 0.05),
            "fan_in_fan_out": config.getoption("fan_in_fan_out", False),
            "target_modules": config.getoption(
                "target_modules",
                _DEFAULT_TARGET_MODULES,
            ),
            "enable_text_adapter": config.getoption("enable_text_adapter", False),
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
    preloaded=False,
):
    weight_path = config.getoption("pretrained_weight_path", None)

    if weight_path is None and not preloaded:
        yield from _flux2_state_dict(pretrained_infos, use_auth_token)
    elif weight_path is not None:
        paths = [weight_path] if isinstance(weight_path, str) else weight_path
        for path in paths:
            yield load_weight(path, use_auth_token=use_auth_token)

    pretrained_lora_weight_path = config.getoption("pretrained_lora_weight_path", None)
    if pretrained_lora_weight_path is not None:
        yield load_weight(
            pretrained_lora_weight_path,
            use_auth_token=use_auth_token,
        )


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
        num_infer_timesteps: Optional[int] = None,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = False,
        target_modules: Optional[Union[List[str], str]] = _DEFAULT_TARGET_MODULES,
        enable_text_adapter: Optional[bool] = False,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "bfloat16",
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
            pretrained_model_path=pretrained_model_path,
            torch_dtype=torch_dtype,
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
            preloaded=model_kwargs["pretrained_model_path"] is not None,
        )
        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict, replace_keys=replace_keys)
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
        guidance_scale: Optional[float] = None,
        enable_cpu_offload: bool = False,
        offload_device: str = "cuda",
    ):
        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            enable_cpu_offload=enable_cpu_offload,
            offload_device=offload_device,
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
        num_infer_timesteps: Optional[int] = None,
        snr_gamma: Optional[float] = 5.0,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = False,
        target_modules: Optional[Union[List[str], str]] = _DEFAULT_TARGET_MODULES,
        enable_text_adapter: Optional[bool] = False,
        enable_transformer_adapter: Optional[bool] = True,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        guidance_scale: Optional[float] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "bfloat16",
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
            pretrained_model_path=pretrained_model_path,
            torch_dtype=torch_dtype,
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
            preloaded=model_kwargs["pretrained_model_path"] is not None,
        )
        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict, replace_keys=replace_keys)
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
        guidance_scale: Optional[float] = None,
        enable_cpu_offload: bool = False,
        offload_device: str = "cuda",
    ):
        outputs = super().generate(
            input_ids=input_ids,
            refer_pixel_values=refer_pixel_values,
            attention_mask=attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            enable_cpu_offload=enable_cpu_offload,
            offload_device=offload_device,
        )
        return DiffusionOutputs(outputs=outputs.images.float())
