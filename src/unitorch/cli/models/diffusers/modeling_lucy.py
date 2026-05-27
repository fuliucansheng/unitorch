# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Optional
from torch import autocast

from unitorch.models.diffusers import (
    LucyForVideoEditingGeneration as _LucyForVideoEditingGeneration,
)
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
)
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_model,
)
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)


def _lucy_model_kwargs(
    config,
    section: str,
    default_pretrained_name: str = "lucy-edit-v1.1-dev",
    pretrained_name: Optional[str] = None,
):
    config.set_default_section(section)
    pretrained_name = pretrained_name or config.getoption(
        "pretrained_name", default_pretrained_name
    )
    pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)
    use_auth_token = config.getoption("use_auth_token", False)

    config_path = config.getoption("config_path", None)
    config_path = pop_value(
        config_path,
        nested_dict_value(pretrained_infos, "transformer", "config"),
    )
    config_path = cached_path(config_path, use_auth_token=use_auth_token)

    text_config_path = config.getoption("text_config_path", None)
    text_config_path = pop_value(
        text_config_path,
        nested_dict_value(pretrained_infos, "text", "config"),
    )
    text_config_path = cached_path(text_config_path, use_auth_token=use_auth_token)

    vae_config_path = config.getoption("vae_config_path", None)
    vae_config_path = pop_value(
        vae_config_path,
        nested_dict_value(pretrained_infos, "vae", "config"),
    )
    vae_config_path = cached_path(vae_config_path, use_auth_token=use_auth_token)

    scheduler_config_path = config.getoption("scheduler_config_path", None)
    scheduler_config_path = pop_value(
        scheduler_config_path,
        nested_dict_value(pretrained_infos, "scheduler"),
    )
    scheduler_config_path = cached_path(
        scheduler_config_path,
        use_auth_token=use_auth_token,
    )

    return {
        "pretrained_infos": pretrained_infos,
        "use_auth_token": use_auth_token,
        "config_path": config_path,
        "text_config_path": text_config_path,
        "vae_config_path": vae_config_path,
        "scheduler_config_path": scheduler_config_path,
        "num_train_timesteps": config.getoption("num_train_timesteps", 1000),
        "num_infer_timesteps": config.getoption("num_infer_timesteps", 50),
        "freeze_vae_encoder": config.getoption("freeze_vae_encoder", True),
        "freeze_text_encoder": config.getoption("freeze_text_encoder", True),
        "snr_gamma": config.getoption("snr_gamma", 5.0),
        "seed": config.getoption("seed", 1123),
        "gradient_checkpointing": config.getoption("gradient_checkpointing", True),
        "expand_timesteps": config.getoption(
            "expand_timesteps",
            nested_dict_value(pretrained_infos, "expand_timesteps") or True,
        ),
    }


def _lucy_state_dict(pretrained_infos, use_auth_token):
    if pretrained_infos is None:
        return None
    return [
        load_weight(
            nested_dict_value(pretrained_infos, "transformer", "weight"),
            prefix_keys={"": "transformer."},
            use_auth_token=use_auth_token,
        ),
        load_weight(
            nested_dict_value(pretrained_infos, "text", "weight"),
            prefix_keys={"": "text."},
            use_auth_token=use_auth_token,
        ),
        load_weight(
            nested_dict_value(pretrained_infos, "vae", "weight"),
            prefix_keys={"": "vae."},
            use_auth_token=use_auth_token,
        ),
    ]


def _load_lucy_loras(inst, config):
    pretrained_lora_names = config.getoption("pretrained_lora_names", None)
    pretrained_lora_weights = config.getoption("pretrained_lora_weights", 1.0)
    pretrained_lora_alphas = config.getoption("pretrained_lora_alphas", 32.0)

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
                pretrained_stable_extensions_infos,
                name,
                "lora",
                "weight",
            )
            for name in pretrained_lora_names
        ]
    else:
        pretrained_lora_weights_path = None

    lora_weights_path = config.getoption(
        "pretrained_lora_weights_path",
        pretrained_lora_weights_path,
    )
    if lora_weights_path is not None:
        inst.load_lora_weights(
            lora_files=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            replace_keys={},
            save_base_state=False,
        )


@register_model("core/model/diffusers/video_editing/lucy", diffusion_model_decorator)
class LucyForVideoEditingGeneration(_LucyForVideoEditingGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
        expand_timesteps: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            expand_timesteps=expand_timesteps,
        )

    @classmethod
    @config_defaults_init("core/model/diffusers/video_editing/lucy")
    def from_config(cls, config, **kwargs):
        section = "core/model/diffusers/video_editing/lucy"
        model_kwargs = _lucy_model_kwargs(config, section)
        pretrained_infos = model_kwargs.pop("pretrained_infos")
        use_auth_token = model_kwargs.pop("use_auth_token")

        inst = cls(**model_kwargs)

        weight_path = config.getoption("pretrained_weight_path", None)
        if weight_path is None:
            state_dict = _lucy_state_dict(pretrained_infos, use_auth_token)
        else:
            state_dict = load_weight(weight_path, use_auth_token=use_auth_token)

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        _load_lucy_loras(inst, config)
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

    @config_defaults_method("core/model/diffusers/video_editing/lucy")
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        guidance_scale: Optional[float] = 5.0,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            refer_pixel_values=refer_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        )

        return DiffusionOutputs(outputs=outputs.frames.float())
