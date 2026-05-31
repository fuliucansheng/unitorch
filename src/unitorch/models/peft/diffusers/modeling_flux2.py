# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Tuple, Union

from peft import LoraConfig

from unitorch.models.diffusers.modeling_flux2 import (
    GenericFlux2Model,
    Flux2ForImageEditingGeneration as _Flux2ForImageEditingGeneration,
    Flux2ForText2ImageGeneration as _Flux2ForText2ImageGeneration,
)
from unitorch.models.peft import PeftCheckpointMixin


class GenericFlux2LoraMixin(PeftCheckpointMixin):
    prefix_keys_in_state_dict = GenericFlux2Model.prefix_keys_in_state_dict
    replace_keys_in_state_dict = {}

    def _add_lora_adapters(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = True,
        target_modules: Optional[Union[List[str], str]] = None,
        enable_text_adapter: bool = True,
        enable_transformer_adapter: bool = True,
    ) -> None:
        target_modules = target_modules or [
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

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        if enable_text_adapter:
            self.text.add_adapter(lora_config)
        if enable_transformer_adapter:
            self.transformer.add_adapter(lora_config)


class Flux2LoraForText2ImageGeneration(
    GenericFlux2LoraMixin, _Flux2ForText2ImageGeneration
):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 50,
        snr_gamma: float = 5.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = True,
        target_modules: Optional[Union[List[str], str]] = None,
        enable_text_adapter: bool = True,
        enable_transformer_adapter: bool = True,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: float = 4.0,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
    ) -> None:
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=True,
            freeze_text_encoder=True,
            snr_gamma=snr_gamma,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            guidance_scale=guidance_scale,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        self._add_lora_adapters(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
        )


class Flux2LoraForImageEditingGeneration(
    GenericFlux2LoraMixin, _Flux2ForImageEditingGeneration
):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: int = 1000,
        num_infer_timesteps: int = 50,
        snr_gamma: float = 5.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = True,
        target_modules: Optional[Union[List[str], str]] = None,
        enable_text_adapter: bool = True,
        enable_transformer_adapter: bool = True,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: float = 4.0,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
    ) -> None:
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=True,
            freeze_text_encoder=True,
            snr_gamma=snr_gamma,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            guidance_scale=guidance_scale,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        self._add_lora_adapters(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            enable_text_adapter=enable_text_adapter,
            enable_transformer_adapter=enable_transformer_adapter,
        )
