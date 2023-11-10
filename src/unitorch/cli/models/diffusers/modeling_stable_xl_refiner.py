# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast

from unitorch.models.diffusers import (
    StableXLRefinerForText2ImageGeneration as _StableXLRefinerForText2ImageGeneration,
    StableXLRefinerForImage2ImageGeneration as _StableXLRefinerForImage2ImageGeneration,
    StableXLRefinerForImageInpainting as _StableXLRefinerForImageInpainting,
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


@register_model(
    "core/model/diffusers/text2image/stable_xl_refiner", diffusion_model_decorator
)
class StableXLRefinerForText2ImageGeneration(_StableXLRefinerForText2ImageGeneration):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        refiner_config_path: Optional[str] = None,
        refiner_text_config_path: Optional[str] = None,
        refiner_text2_config_path: Optional[str] = None,
        refiner_vae_config_path: Optional[str] = None,
        refiner_scheduler_config_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            refiner_config_path=refiner_config_path,
            refiner_text_config_path=refiner_text_config_path,
            refiner_text2_config_path=refiner_text2_config_path,
            refiner_vae_config_path=refiner_vae_config_path,
            refiner_scheduler_config_path=refiner_scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            seed=seed,
        )

    @classmethod
    @add_default_section_for_init("core/model/diffusers/text2image/stable_xl_refiner")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("core/model/diffusers/text2image/stable_xl_refiner")
        pretrained_name = config.getoption(
            "pretrained_name", "stable-xl-base-refiner-1.0"
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

        text2_config_path = config.getoption("text2_config_path", None)
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrain_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

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

        refiner_config_path = config.getoption("refiner_config_path", None)
        refiner_config_path = pop_value(
            refiner_config_path,
            nested_dict_value(pretrain_infos, "refiner_unet", "config"),
            check_none=False,
        )
        refiner_config_path = (
            cached_path(refiner_config_path)
            if refiner_config_path is not None
            else None
        )

        refiner_text_config_path = config.getoption("refiner_text_config_path", None)
        refiner_text_config_path = pop_value(
            refiner_text_config_path,
            nested_dict_value(pretrain_infos, "refiner_text", "config"),
            check_none=False,
        )
        refiner_text_config_path = (
            cached_path(refiner_text_config_path)
            if refiner_text_config_path is not None
            else None
        )

        refiner_text2_config_path = config.getoption("refiner_text2_config_path", None)
        refiner_text2_config_path = pop_value(
            refiner_text2_config_path,
            nested_dict_value(pretrain_infos, "refiner_text2", "config"),
            check_none=False,
        )
        refiner_text2_config_path = (
            cached_path(refiner_text2_config_path)
            if refiner_text2_config_path is not None
            else None
        )

        refiner_vae_config_path = config.getoption("refiner_vae_config_path", None)
        refiner_vae_config_path = pop_value(
            refiner_vae_config_path,
            nested_dict_value(pretrain_infos, "refiner_vae", "config"),
        )
        refiner_vae_config_path = cached_path(refiner_vae_config_path)

        refiner_scheduler_config_path = config.getoption(
            "refiner_scheduler_config_path", None
        )
        refiner_scheduler_config_path = pop_value(
            refiner_scheduler_config_path,
            nested_dict_value(pretrain_infos, "refiner_scheduler"),
            check_none=False,
        )
        refiner_scheduler_config_path = (
            cached_path(refiner_scheduler_config_path)
            if refiner_scheduler_config_path is not None
            else None
        )

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
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            refiner_config_path=refiner_config_path,
            refiner_text_config_path=refiner_text_config_path,
            refiner_text2_config_path=refiner_text2_config_path,
            refiner_vae_config_path=refiner_vae_config_path,
            refiner_scheduler_config_path=refiner_scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrain_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrain_infos, "unet", "weight"),
                    prefix_keys={"": "unet."},
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    nested_dict_value(pretrain_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]
            if nested_dict_value(pretrain_infos, "refiner_unet", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_unet", "weight"),
                        prefix_keys={"": "refiner_unet."},
                    )
                )
            if nested_dict_value(pretrain_infos, "refiner_text", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_text", "weight"),
                        prefix_keys={"": "refiner_text."},
                    )
                )
            if nested_dict_value(pretrain_infos, "refiner_text2", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_text2", "weight"),
                        prefix_keys={"": "refiner_text2."},
                    )
                )
            if nested_dict_value(pretrain_infos, "refiner_vae", "weight") is not None:
                state_dict.append(
                    load_weight(
                        nested_dict_value(pretrain_infos, "refiner_vae", "weight"),
                        prefix_keys={"": "refiner_vae."},
                    )
                )

        inst.from_pretrained(weight_path, state_dict=state_dict)
        return inst

    # @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        add_time_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
    ):
        loss = super().forward(
            input_ids=input_ids,
            input2_ids=input2_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            add_time_ids=add_time_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
        )
        return LossOutputs(loss=loss)

    @add_default_section_for_function(
        "core/model/diffusers/text2image/stable_xl_refiner"
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        refiner_input_ids: Optional[torch.Tensor] = None,
        refiner_input2_ids: Optional[torch.Tensor] = None,
        refiner_negative_input_ids: Optional[torch.Tensor] = None,
        refiner_negative_input2_ids: Optional[torch.Tensor] = None,
        refiner_attention_mask: Optional[torch.Tensor] = None,
        refiner_attention2_mask: Optional[torch.Tensor] = None,
        refiner_negative_attention_mask: Optional[torch.Tensor] = None,
        refiner_negative_attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        high_noise_frac: Optional[float] = 0.8,
        guidance_scale: Optional[float] = 5.0,
    ):
        outputs = super().generate(
            input_ids=input_ids,
            input2_ids=input2_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
            refiner_input_ids=refiner_input_ids,
            refiner_input2_ids=refiner_input2_ids,
            refiner_negative_input_ids=refiner_negative_input_ids,
            refiner_negative_input2_ids=refiner_negative_input2_ids,
            refiner_attention_mask=refiner_attention_mask,
            refiner_attention2_mask=refiner_attention2_mask,
            refiner_negative_attention_mask=refiner_negative_attention_mask,
            refiner_negative_attention2_mask=refiner_negative_attention2_mask,
            height=height,
            width=width,
            high_noise_frac=high_noise_frac,
            guidance_scale=guidance_scale,
        )

        return DiffusionOutputs(outputs=outputs.images)
