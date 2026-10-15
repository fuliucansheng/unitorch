# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import List, Optional, Tuple, Union

import torch
from peft import LoraConfig

from unitorch.models.diffusers.modeling_flux2 import (
    GenericFlux2Model,
    Flux2ForImageEditingGeneration as _Flux2ForImageEditingGeneration,
    Flux2ForText2ImageGeneration as _Flux2ForText2ImageGeneration,
)
from unitorch.models.peft import PeftCheckpointMixin


def _checkpoint_lora_block(module, *args):
    """Recompute a frozen-base block while ZeRO gathers its adapter weights."""
    if not args[0].requires_grad:
        args = (args[0].detach().requires_grad_(True), *args[1:])
    return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=True)


class GenericFlux2LoraMixin(PeftCheckpointMixin):
    prefix_keys_in_state_dict = GenericFlux2Model.prefix_keys_in_state_dict
    replace_keys_in_state_dict = {}

    def from_checkpoint(self, ckpt_dir, weight_name=None, **kwargs):
        """Validate adapter coverage and gather ZeRO partitions while restoring."""
        return GenericFlux2Model.from_checkpoint(self, ckpt_dir, weight_name, **kwargs)

    def checkpoint_state_dict(self, state_dict=None):
        """Export adapter tensors, excluding ZeRO's frozen conditioning buffers."""
        state_dict = self.state_dict() if state_dict is None else state_dict
        return {key: value for key, value in state_dict.items()
                if any(part in key for part in self.modules_to_save_checkpoints)}

    def _add_lora_adapters(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        target_modules: Optional[Union[List[str], str]] = None,
        enable_text_adapter: bool = False,
        enable_transformer_adapter: bool = True,
    ) -> None:
        transformer_targets = target_modules or [
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_out.0",
            "to_add_out",
            "to_qkv_mlp_proj",
            *[f"single_transformer_blocks.{index}.attn.to_out"
              for index in range(len(self.transformer.single_transformer_blocks))],
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
            target_modules=transformer_targets,
        )
        if enable_text_adapter:
            text_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, init_lora_weights="gaussian",
                lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out,
                target_modules=target_modules or ["q_proj", "k_proj", "v_proj"],
            )
            self.text.add_adapter(text_config)
        if enable_transformer_adapter:
            self.transformer.add_adapter(lora_config)
        for module in self.modules():
            if not hasattr(module, "lora_A"):
                continue
            for name in module.lora_A:
                parameters = [module.lora_A[name].weight, module.lora_B[name].weight]
                if any(hasattr(parameter, "ds_id") for parameter in parameters):
                    import deepspeed

                    with deepspeed.zero.GatheredParameters(parameters, modifier_rank=0):
                        if deepspeed.comm.get_rank() == 0:
                            module.reset_lora_parameters(name, init_lora_weights="gaussian")
        # Adam updates must not disappear in BF16 rounding of small adapters.
        # ZeRO owns its partition dtype and fp32 optimizer master weights.
        for parameter in self.parameters():
            if parameter.requires_grad and not hasattr(parameter, "ds_id"):
                parameter.data = parameter.data.float()
        if self.transformer.gradient_checkpointing and any(
            hasattr(parameter, "ds_id") for parameter in self.transformer.parameters()
        ):
            # Non-reentrant checkpointing retains metadata for frozen tensors
            # that ZeRO releases to empty placeholders during recomputation.
            self.transformer.enable_gradient_checkpointing(_checkpoint_lora_block)
        self.train(self.training)


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
        num_infer_timesteps: Optional[int] = None,
        snr_gamma: float = 5.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        target_modules: Optional[Union[List[str], str]] = None,
        enable_text_adapter: bool = False,
        enable_transformer_adapter: bool = True,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: Optional[float] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
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
            pretrained_model_path=pretrained_model_path,
            torch_dtype=torch_dtype,
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
        num_infer_timesteps: Optional[int] = None,
        snr_gamma: float = 5.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fan_in_fan_out: bool = False,
        target_modules: Optional[Union[List[str], str]] = None,
        enable_text_adapter: bool = False,
        enable_transformer_adapter: bool = True,
        seed: int = 1123,
        gradient_checkpointing: bool = True,
        guidance_scale: Optional[float] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
        pretrained_model_path: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = torch.bfloat16,
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
            pretrained_model_path=pretrained_model_path,
            torch_dtype=torch_dtype,
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
