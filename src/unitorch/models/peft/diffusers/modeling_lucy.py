# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Union

from peft import LoraConfig

from unitorch.models.diffusers.modeling_lucy import (
    LucyForVideoEditingGeneration as _LucyForVideoEditingGeneration,
)
from unitorch.models.peft import PeftCheckpointMixin, add_adapter_compat


_DEFAULT_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
]


class GenericLucyLoraMixin(PeftCheckpointMixin):
    text_target_modules = ["q", "k", "v", "o"]

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
        target_modules = target_modules or _DEFAULT_TARGET_MODULES

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False

        transformer_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        if enable_text_adapter:
            text_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                lora_dropout=lora_dropout,
                fan_in_fan_out=False,
                target_modules=self.text_target_modules,
            )
            self.text = add_adapter_compat(self.text, text_lora_config)
        if enable_transformer_adapter:
            self.transformer = add_adapter_compat(
                self.transformer,
                transformer_lora_config,
            )


class LucyLoraForVideoEditingGeneration(
    GenericLucyLoraMixin, _LucyForVideoEditingGeneration
):
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
        expand_timesteps: Optional[bool] = True,
    ):
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
            expand_timesteps=expand_timesteps,
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
