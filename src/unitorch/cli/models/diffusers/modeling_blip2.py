# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast

from unitorch.models.diffusers import (
    Blip2ForText2ImageGeneration as _Blip2ForText2ImageGeneration,
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
from unitorch.cli.models.diffusers import pretrained_diffusers_infos, load_weight


@register_model("core/model/diffusers/text2image/blip2", diffusion_model_decorator)
class Blip2ForText2ImageGeneration(_Blip2ForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        qformer_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        prior_loss_weight: Optional[float] = 1.0,
        lora_r: Optional[int] = None,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            qformer_config_path=qformer_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            prior_loss_weight=prior_loss_weight,
            lora_r=lora_r,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/text2image/blip2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/text2image/blip2")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-v1.5-blipdiffuion"
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

        qformer_config_path = config.getoption("qformer_config_path", None)
        qformer_config_path = pop_value(
            qformer_config_path,
            nested_dict_value(pretrain_infos, "qformer", "config"),
        )
        qformer_config_path = cached_path(qformer_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

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
        snr_gamma = config.getoption("snr_gamma", 5.0)
        prior_loss_weight = config.getoption("prior_loss_weight", 1.0)
        lora_r = config.getoption("lora_r", None)
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            qformer_config_path=qformer_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            prior_loss_weight=prior_loss_weight,
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
                        "\.query\.": ".to_q.",
                        "\.key\.": ".to_k.",
                        "\.value\.": ".to_v.",
                        "\.proj_attn\.": ".to_out.0.",
                    },
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "text", "weight"),
                    replace_keys={
                        "\.query\.": ".to_q.",
                        "\.key\.": ".to_k.",
                        "\.value\.": ".to_v.",
                        "\.proj_attn\.": ".to_out.0.",
                    },
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "vae", "weight"),
                    replace_keys={
                        "\.query\.": ".to_q.",
                        "\.key\.": ".to_k.",
                        "\.value\.": ".to_v.",
                        "\.proj_attn\.": ".to_out.0.",
                    },
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "qformer", "weight"),
                    prefix_keys={"": "qformer."},
                ),
            ]

        inst.from_pretrained(weight_path, state_dict=state_dict)
        return inst

    @autocast()
    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("core/model/diffusers/text2image/blip2")
    @autocast()
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        refer_input_ids: torch.Tensor,
        refer_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        refer_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: Optional[float] = 7.5,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            refer_input_ids=refer_input_ids,
            refer_pixel_values=refer_pixel_values,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            refer_attention_mask=refer_attention_mask,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )

        return DiffusionOutputs(outputs=outputs.images)
